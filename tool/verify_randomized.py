"""Seeded differential checks for dynamic-shape graphs using ONNX Runtime."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h
import onnxruntime as ort

from verify_models import exchange


def main():
    rng = np.random.default_rng(314)
    info = lambda name, shape: h.make_tensor_value_info(name, T.FLOAT, shape)
    x, w, b = (
        info("x", ["batch", "seq", "hidden"]),
        info("w", ["hidden", "out"]),
        info("b", ["out"]),
    )
    graph = h.make_graph(
        [
            h.make_node("MatMul", ["x", "w"], ["product"]),
            h.make_node("Add", ["product", "b"], ["logits"]),
            h.make_node("Softmax", ["logits"], ["probs"], axis=-1),
        ],
        "projection",
        [x, w, b],
        [
            info("logits", ["batch", "seq", "out"]),
            info("probs", ["batch", "seq", "out"]),
        ],
    )
    model = h.make_model(graph, ir_version=14, opset_imports=[h.make_opsetid("", 28)])
    onnx.checker.check_model(model, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    oracle = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        exe = folder / "compare"
        path = folder / "model.onnx"
        onnx.save(model, path)
        subprocess.run(
            ["dart", "compile", "exe", "tool/compare_runner.dart", "-o", str(exe)],
            check=True,
        )
        with subprocess.Popen(
            [str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        ) as proc:
            try:
                for _ in range(50):
                    batch, seq, hidden, out = rng.integers(1, 17, size=4)
                    feed = {
                        k: rng.normal(size=s).astype("f")
                        for k, s in [
                            ("x", (batch, seq, hidden)),
                            ("w", (hidden, out)),
                            ("b", (out,)),
                        ]
                    }
                    expected = dict(zip(["logits", "probs"], oracle.run(None, feed)))
                    exchange(proc, folder, path, feed, expected, predictions=False)
            finally:
                proc.stdin.close()
    print("50 seeded dynamic-shape cases passed.")


if __name__ == "__main__":
    main()
