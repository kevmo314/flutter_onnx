"""Exercise upstream standard-domain node cases at opset 28 and record gaps."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import tempfile
import warnings

import onnx
from onnx.backend.test.case.node import collect_testcases
from verify_models import exchange


def assess(binary, folder):
    if onnx.__version__ != '1.23.2':
        raise ValueError('Use onnx==1.23.2 for the opset 28 baseline')
    schemas={s.name:s for s in onnx.defs.get_all_schemas() if not s.domain}
    results={name:{} for name in sorted(schemas)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        cases=collect_testcases()
    with subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True) as proc:
        try:
            for case in cases:
                model=case.model
                # Expanded function bodies exercise a different graph, and are
                # covered separately from the operator's own unexpanded cases.
                if '_expanded' in case.name or len(model.graph.node)!=1: continue
                node=model.graph.node[0]
                if node.domain or node.op_type not in schemas: continue
                record=results[node.op_type]
                try:
                    imported=next(o for o in model.opset_import if o.domain=='')
                    if imported.version!=28:
                        # Use the original fixture unchanged if its checked graph
                        # is valid at 28; otherwise use the official version adapter.
                        old=imported.version
                        imported.version=28
                        try: onnx.checker.check_model(model,full_check=True)
                        except Exception:
                            imported.version=old
                            model=onnx.version_converter.convert_version(model,28)
                    onnx.checker.check_model(model,full_check=True)
                    path=folder/(case.name+'.onnx')
                    onnx.save(model,path)
                    for inputs,outputs in case.data_sets:
                        inputs=[onnx.numpy_helper.to_array(v) if isinstance(v,onnx.TensorProto) else v for v in inputs]
                        outputs=[onnx.numpy_helper.to_array(v) if isinstance(v,onnx.TensorProto) else v for v in outputs]
                        feeds=dict(zip([i.name for i in model.graph.input],inputs,strict=True))
                        expected=dict(zip([o.name for o in model.graph.output],outputs,strict=True))
                        exchange(proc,folder,path,feeds,expected,predictions=False,
                                 rtol=case.rtol,atol=case.atol)
                    record[case.name]='pass'
                except Exception as exc:
                    reason=str(exc).strip().splitlines()
                    record[case.name]=(reason[0] if reason else type(exc).__name__)[:180]
        finally: proc.stdin.close()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--baseline',type=Path,help='Fail if a previously passing case regresses.')
    parser.add_argument('--strict',action='store_true',help='Fail on any unverified operator or failed case.')
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        binary=args.runner.resolve() if args.runner else folder/'compare'
        if not args.runner: subprocess.run(['dart','compile','exe','tool/compare_runner.dart','-o',str(binary)],check=True)
        results=assess(binary,folder)
    if args.report:
        args.report.write_text(json.dumps(results,indent=2,sort_keys=True)+'\n')
    counts=collections.Counter(status=='pass' for cases in results.values() for status in cases.values())
    complete=sum(bool(cases) and all(v=='pass' for v in cases.values()) for cases in results.values())
    print(f"{counts[True]}/{sum(counts.values())} standard-domain node cases pass; "
          f"{complete}/{len(results)} operators pass every available case with at least one case.")
    if args.baseline:
        expected=set(args.baseline.read_text().splitlines())
        actual={name for cases in results.values() for name,status in cases.items() if status=='pass'}
        if expected-actual: raise SystemExit('Regressions: '+', '.join(sorted(expected-actual)))
    if args.strict and complete!=len(results): raise SystemExit(1)


if __name__=='__main__': main()
