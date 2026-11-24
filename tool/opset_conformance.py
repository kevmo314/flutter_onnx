"""Exercise upstream standard-domain node cases at opset 28 and record gaps."""

import argparse
import collections
import contextlib
from device_runner import DeviceRunner
import json
from pathlib import Path
import subprocess
import tempfile
import warnings

import onnx
import numpy as np
from onnx.backend.test.case.node import collect_testcases
from verify_models import exchange
import verify_supplemental
from freeze_dft import stored_datasets


def check_staged_model(model):
    try:
        onnx.checker.check_model(model, full_check=True)
    except onnx.shape_inference.InferenceError:
        # Some upstream function bodies reference a default attribute in a
        # Constant. Validate the original model and infer shapes on its
        # semantically equivalent explicit-default form. Execute the original.
        onnx.checker.check_model(model)
        canonical = onnx.ModelProto()
        canonical.CopyFrom(model)
        changed = False
        for node in canonical.graph.node:
            if node.domain:
                continue
            schema = onnx.defs.get_schema(node.op_type, 28)
            if not (schema.has_function or schema.has_context_dependent_function):
                continue
            present = {a.name for a in node.attribute}
            for name, attr in schema.attributes.items():
                if name not in present and attr.default_value.type:
                    node.attribute.append(attr.default_value)
                    changed = True
        if not changed:
            raise
        onnx.checker.check_model(canonical, full_check=True)


def check_bernoulli_distribution(proc, folder, model, case):
    # ONNX specifies the distribution and output dtype, leaving the PRNG
    # implementation-dependent. Also check the fixture literally elsewhere.
    original, expected = case.data_sets[0]
    probabilities = np.r_[0, original[0].flatten(), 1].astype(original[0].dtype)
    repetitions = 8192
    batch = np.tile(probabilities, (repetitions, 1))
    staged = onnx.ModelProto()
    staged.CopyFrom(model)
    for value in (staged.graph.input[0], staged.graph.output[0]):
        tensor = value.type.tensor_type
        value.CopyFrom(
            onnx.helper.make_tensor_value_info(
                value.name, tensor.elem_type, list(batch.shape)
            )
        )
    check_staged_model(staged)
    path = folder / (case.name + "-distribution.onnx")
    onnx.save(staged, path)

    def distribution(_, actual):
        assert np.all((actual == 0) | (actual == 1)), "Nonbinary Bernoulli draw"
        assert np.all(actual[:, 0] == 0) and np.all(actual[:, -1] == 1), (
            "Wrong deterministic endpoint"
        )
        # For 8192 independent draws the two-sided Hoeffding bound for
        # a 0.05 error is below 4e-18 per input probability.
        np.testing.assert_allclose(
            actual.mean(axis=0), probabilities, rtol=0, atol=0.05
        )

    exchange(
        proc,
        folder,
        path,
        {staged.graph.input[0].name: batch},
        {staged.graph.output[0].name: np.empty(batch.shape, dtype=expected[0].dtype)},
        predictions=False,
        check_distribution=distribution,
    )


def assess(
    binary,
    folder,
    selected=None,
    verbose=False,
    browser=None,
    device=None,
    supplemental=False,
):
    if onnx.__version__ != "1.23.2":
        raise ValueError("Use onnx==1.23.2 for the opset 28 baseline")
    schemas = {s.name: s for s in onnx.defs.get_all_schemas() if not s.domain}
    results = {name: {} for name in sorted(schemas)}
    distributions = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cases = collect_testcases()
    transport = (
        contextlib.closing(DeviceRunner(folder, binary, browser, device, timeout=90))
        if browser or device
        else subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
    )
    with transport as proc:
        try:
            for case in cases:
                model = case.model
                # Expanded function bodies exercise a different graph, and are
                # covered separately from the operator's own unexpanded cases.
                if "_expanded" in case.name or len(model.graph.node) != 1:
                    continue
                if selected and case.name not in selected:
                    continue
                node = model.graph.node[0]
                if node.domain or node.op_type not in schemas:
                    continue
                record = results[node.op_type]
                try:
                    # ONNX regenerates expected FFT values with the host NumPy.
                    # Send the identical upstream vectors to each target.
                    datasets = (
                        stored_datasets(case)
                        if node.op_type == "DFT"
                        else case.data_sets
                    )
                    imported = next(o for o in model.opset_import if o.domain == "")
                    legacy_scan = node.op_type == "Scan" and imported.version <= 8
                    if imported.version != 28:
                        # Use the original fixture unchanged if its checked graph
                        # is valid at 28; otherwise use the official version adapter.
                        old = imported.version
                        imported.version = 28
                        try:
                            if legacy_scan:
                                # Scan-8 used an implicit leading batch dimension.
                                # A version-tag edit cannot translate that contract.
                                raise ValueError("Scan-8 requires an adapter")
                            check_staged_model(model)
                        except Exception:
                            imported.version = old
                            model = onnx.version_converter.convert_version(model, 28)
                    check_staged_model(model)
                    path = folder / (case.name + ".onnx")
                    onnx.save(model, path)
                    if node.op_type == "Bernoulli":
                        try:
                            check_bernoulli_distribution(proc, folder, model, case)
                            distributions[case.name] = "pass"
                        except TimeoutError:
                            raise
                        except Exception as error:
                            distributions[case.name] = str(error)
                    if legacy_scan and not node.input[0]:
                        # The official Scan-8 adapter removes the leading batch
                        # axis from the graph. Apply that change to every batch
                        # of the stored input and reference output as well.
                        datasets = [
                            ([v[i] for v in xs], [v[i] for v in ys])
                            for xs, ys in datasets
                            for i in range(xs[0].shape[0])
                        ]
                    for inputs, outputs in datasets:
                        inputs = [
                            onnx.numpy_helper.to_array(v)
                            if isinstance(v, onnx.TensorProto)
                            else v
                            for v in inputs
                        ]
                        outputs = [
                            onnx.numpy_helper.to_array(v)
                            if isinstance(v, onnx.TensorProto)
                            else v
                            for v in outputs
                        ]
                        feeds = dict(
                            zip(
                                [i.name for i in model.graph.input], inputs, strict=True
                            )
                        )
                        expected = dict(
                            zip(
                                [o.name for o in model.graph.output],
                                outputs,
                                strict=True,
                            )
                        )
                        exchange(
                            proc,
                            folder,
                            path,
                            feeds,
                            expected,
                            predictions=False,
                            rtol=case.rtol,
                            atol=case.atol,
                        )
                    record[case.name] = "pass"
                except TimeoutError:
                    raise
                except Exception as exc:
                    if verbose or node.op_type != "Bernoulli":
                        print(case.name, str(exc), flush=True)
                    reason = str(exc).strip().splitlines()
                    record[case.name] = (reason[0] if reason else type(exc).__name__)[
                        :180
                    ]
            if supplemental:
                verify_supplemental.check(proc, folder)
        finally:
            if not browser and not device:
                proc.stdin.close()
    return results, distributions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", help="Run an individual upstream case (repeatable)."
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--runner", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--browser", help="Run the fixtures in this Chromium executable.")
    mode.add_argument(
        "--device",
        help="Run fixtures on the installed Android verification app or iOS simulator UDID.",
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--baseline", type=Path, help="Fail if a previously passing case regresses."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any unverified operator or failed case.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        binary = args.runner.resolve() if args.runner else folder / "compare"
        if args.browser and not args.runner:
            binary = folder / "client.js"
            subprocess.run(
                [
                    "dart",
                    "compile",
                    "js",
                    "-O2",
                    "tool/model_client.dart",
                    "-o",
                    str(binary),
                ],
                check=True,
            )
        elif not args.runner and not args.device:
            subprocess.run(
                [
                    "dart",
                    "compile",
                    "exe",
                    "tool/compare_runner.dart",
                    "-o",
                    str(binary),
                ],
                check=True,
            )
        results, distributions = assess(
            binary,
            folder,
            args.case,
            args.verbose,
            args.browser,
            args.device,
            args.strict,
        )
    if args.report:
        args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    counts = collections.Counter(
        status == "pass" for cases in results.values() for status in cases.values()
    )
    complete = sum(
        bool(cases) and all(v == "pass" for v in cases.values())
        for cases in results.values()
    )
    print(
        f"{counts[True]}/{sum(counts.values())} standard-domain node cases pass; "
        f"{complete}/{len(results)} operators pass every available case with at least one case."
    )
    random_names = set(results.get("Bernoulli", {}))
    random_pass = {k for k, v in distributions.items() if v == "pass"}
    deterministic_failures = {
        name
        for op, cases in results.items()
        if op != "Bernoulli"
        for name, v in cases.items()
        if v != "pass"
    }
    unmet = deterministic_failures | (random_names - random_pass)
    print(
        f"{sum(counts.values()) - len(unmet)}/{sum(counts.values())} cases meet operator-specific checks, including {len(random_pass)} Bernoulli distribution checks."
    )
    if args.report:
        args.report.with_suffix(".stochastic.json").write_text(
            json.dumps(distributions, indent=2, sort_keys=True) + "\n"
        )
    if args.baseline and random_names - random_pass:
        raise SystemExit("Bernoulli distribution check failed: " + str(distributions))
    if args.baseline:
        expected = set(args.baseline.read_text().splitlines())
        actual = {
            name
            for cases in results.values()
            for name, status in cases.items()
            if status == "pass"
        }
        if expected - actual:
            raise SystemExit("Regressions: " + ", ".join(sorted(expected - actual)))
    if args.strict:
        # These schemas have no unexpanded fixture in the pinned ONNX release.
        # The supplemental suite executes each one on a schema-checked graph.
        uncovered = {
            name for name, cases in results.items() if not cases
        } - verify_supplemental.OPERATORS
        if unmet or uncovered:
            raise SystemExit(1)
        print(
            f"Strict conformance passed across all {len(results)} standard operators."
        )


if __name__ == "__main__":
    main()
