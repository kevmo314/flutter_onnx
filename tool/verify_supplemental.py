"""Check the 14 opset-28 schemas without an upstream single-node fixture."""

import base64
import shutil
import numpy as np
import onnx
from onnx import helper
from generate_supplemental import models
from verify_models import exchange

OPERATORS = {
    "ConcatFromSequence",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceErase",
    "SequenceLength",
    "Optional",
    "GlobalLpPool",
    "MaxRoiPool",
    "Multinomial",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
}


def check_fourier(proc, folder):
    rng = np.random.default_rng(314)
    for dtype, tensor_type in (
        (np.float32, onnx.TensorProto.FLOAT),
        (np.float64, onnx.TensorProto.DOUBLE),
    ):
        info = lambda name: helper.make_tensor_value_info(name, tensor_type, [None, 2])
        for inverse in (False, True):
            model = helper.make_model(
                helper.make_graph(
                    [helper.make_node("DFT", ["x"], ["y"], inverse=int(inverse))],
                    "complex_fourier",
                    [info("x")],
                    [info("y")],
                ),
                opset_imports=[helper.make_opsetid("", 28)],
                ir_version=14,
            )
            onnx.checker.check_model(model, full_check=True)
            path = folder / f"fourier-{tensor_type}-{inverse}.onnx"
            onnx.save(model, path)
            for length in (1, 5, 10, 16):
                x = rng.normal(size=(length, 2)).astype(dtype)
                z = x[:, 0].astype(np.complex128) + 1j * x[:, 1].astype(np.float64)
                y = np.fft.ifft(z) if inverse else np.fft.fft(z)
                expected = np.stack((y.real, y.imag), axis=1).astype(dtype)
                tolerance = 2e-6 if dtype == np.float32 else 1e-12
                exchange(
                    proc,
                    folder,
                    path,
                    {"x": x},
                    {"y": expected},
                    predictions=False,
                    rtol=tolerance,
                    atol=tolerance,
                )


def check(proc, folder):
    check_fourier(proc, folder)

    def model(name):
        path = folder / (name + "-additional.onnx")
        path.write_bytes(base64.b64decode(models[name]))
        return path

    a = np.array([1, 3], dtype=np.float32)
    b = np.array([5, 7], dtype=np.float32)
    exchange(
        proc,
        folder,
        model("containers"),
        {"a": a, "b": b, "idx": np.array(1, np.int64)},
        {"at": a, "present": b, "joined": np.r_[a, a], "length": np.array(1, np.int64)},
        predictions=False,
    )
    feature = np.arange(-8, 8, dtype=np.float32).reshape(1, 1, 4, 4)
    exchange(
        proc,
        folder,
        model("pool"),
        {"x": feature, "rois": np.array([[0, 0, 0, 3, 3]], np.float32)},
        {
            "roi": np.array([[[[-3, -1], [5, 7]]]], np.float32),
            "norm": np.sqrt(np.sum(feature**2, axis=(2, 3), keepdims=True)),
        },
        predictions=False,
    )
    count = 12000
    feeds = {
        "x": np.full(count, 0.25, np.float32),
        "logits": np.log(np.array([[1, 3]], np.float32)),
        "ratio": np.array(0.5, np.float32),
        "training": np.array(True),
    }
    expected = {
        name: np.empty(count, np.float32)
        for name in [
            "RandomUniform",
            "RandomUniformLike",
            "RandomNormal",
            "RandomNormalLike",
            "coin",
            "drop",
        ]
    }
    expected.update(
        mask=np.empty(count, np.bool_), classes=np.empty((1, count), np.int32)
    )

    def samples(name, values):
        if name.startswith("RandomNormal"):
            assert abs(values.mean()) < 0.05 and abs(np.mean(values**2) - 1) < 0.07, (
                name
            )
        elif name.startswith("RandomUniform"):
            assert (
                np.all((values >= 0) & (values < 1)) and abs(values.mean() - 0.5) < 0.02
            ), name
        elif name in ("classes", "coin", "mask"):
            assert np.all((values == 0) | (values == 1)), name
            assert (
                abs(values.mean() - {"classes": 0.75, "coin": 0.25, "mask": 0.5}[name])
                < 0.03
            ), name

    path = model("random")

    def draw(path):
        captured = dict.fromkeys(expected)
        exchange(
            proc,
            folder,
            path,
            feeds,
            expected,
            predictions=False,
            capture=captured,
            check_distribution=samples,
        )
        np.testing.assert_array_equal(captured["drop"], captured["mask"] * 0.5)
        return captured

    first, next_draw = draw(path), draw(path)
    copy = folder / "random-additional-replay.onnx"
    shutil.copyfile(path, copy)
    replay = draw(copy)
    for name in first:
        np.testing.assert_array_equal(first[name], replay[name], err_msg=name)
        assert np.any(first[name] != next_draw[name]), name
