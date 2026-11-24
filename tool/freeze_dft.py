"""Freeze ONNX's generated DFT vectors for bit-identical cross-platform tests."""

import hashlib
from pathlib import Path
import platform
import warnings
import numpy as np
import onnx
from onnx.backend.test.case.node import collect_testcases

VECTORS = Path(__file__).with_name("fixtures") / "dft-opset28.npz"


def graph_digest(case):
    return hashlib.sha256(case.model.SerializeToString(deterministic=True)).hexdigest()


def freeze():
    if (onnx.__version__, np.__version__, platform.machine(), platform.system()) != (
        "1.23.2",
        "2.3.5",
        "x86_64",
        "Linux",
    ):
        raise SystemExit("Regenerate with ONNX 1.23.2 and NumPy 2.3.5 on x86_64 Linux.")
    arrays = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cases = collect_testcases()
    for case in cases:
        nodes = case.model.graph.node
        if len(nodes) != 1 or nodes[0].op_type != "DFT" or "_expanded" in case.name:
            continue
        arrays[case.name + "_graph"] = np.array(graph_digest(case))
        for i, (inputs, outputs) in enumerate(case.data_sets):
            for side, values in (("in", inputs), ("out", outputs)):
                for j, value in enumerate(values):
                    arrays[f"{case.name}_{i}_{side}_{j}"] = value
    VECTORS.parent.mkdir(exist_ok=True)
    np.savez_compressed(VECTORS, **arrays)


def stored_datasets(case):
    with np.load(VECTORS, allow_pickle=False) as arrays:
        assert arrays[case.name + "_graph"].item() == graph_digest(case), case.name
        return [
            tuple(
                [arrays[f"{case.name}_{i}_{side}_{j}"] for j in range(len(values))]
                for side, values in (("in", inputs), ("out", outputs))
            )
            for i, (inputs, outputs) in enumerate(case.data_sets)
        ]


if __name__ == "__main__":
    freeze()
