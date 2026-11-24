"""Compare unmodified ONNX Model Zoo and Qwen exports against Python ONNX Runtime.

Install numpy and onnxruntime. Run from the repo root, e.g.
python tool/verify_models.py --cache /tmp/onnx-models resnet18 qwen
"""

import argparse
import contextlib
from device_runner import DeviceRunner
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import onnx
import numpy as np
import onnxruntime as ort

ZOO = "https://media.githubusercontent.com/media/onnx/models/c5cae0f1942c8d7727372c7b1f6b485aa39e4421/Computer_Vision"
MODELS = {
    "squeezenet1_1": (
        "torch_hub",
        "fbb553444cab298bcac72cf623694e277dfe11d78662108f91ff46b6acb2cc5f",
    ),
    "mobilenetv2_050": (
        "timm",
        "192f09d45d96c3a4ca8ea271613e2b48c37ac15a93396fde3a4d7b2ea0aad36b",
    ),
    "resnet18": (
        "torch_hub",
        "eef70310eb26bdbc70babb290ee3239599ea9aed406bade711e3d2693b9d20e8",
    ),
    "efficientnet_b0": (
        "timm",
        "e76596a2b9e27c7c734c38550859105b43fec926f13447a84dad175eb994068a",
    ),
    "densenet121": (
        "torch_hub",
        "9299d867cac3e0bf40803ac737e3fdaf3cb5976ee65c82961c211a3d8be3bbdd",
    ),
    "mobilenetv3_small_050": (
        "timm",
        "ce71d83541479fb12b543415aa620d24c518c8ee0cad60990140f05c6ea4e462",
    ),
}
QWEN_URL = "https://huggingface.co/onnx-community/Qwen2.5-0.5B-Instruct/resolve/22942cb7d7ba4cc81bb4673549ca4d4614469b5e/onnx/model.onnx"


def obtain(name, cache):
    if name == "qwen3_q4":
        url = "https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/resolve/79346d370dfae34801d26993fda9ae2372900fd0/onnx/model_q4.onnx"
        digest = "d43d836fc5e240df9013733ccd214972c5d21bd9ec47e574e4f1e359cf90aed0"
    elif name == "qwen_q4":
        url, digest = (
            QWEN_URL.replace("/model.onnx", "/model_q4.onnx"),
            "09235a3b1c135cd04ef570e5053b5c079e028078a2cc5f76ba6a251e91bf3296",
        )
    elif name == "qwen":
        url, digest = (
            QWEN_URL,
            "373b18ccddcf62f765516663f957866ce8a63e885abd1d2878cc464c5f7e25d7",
        )
    else:
        source, digest = MODELS[name]
        url = f"{ZOO}/{name}_Opset17_{source}/{name}_Opset17.onnx"
    path = cache / (name + ".onnx")
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".download")
        urllib.request.urlretrieve(url, temp)
        temp.replace(path)
    with path.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"{path}: incorrect SHA-256 {actual}")
    return path


def exchange(
    proc,
    folder,
    model,
    feeds,
    expected,
    cache_names=None,
    *,
    external_data=None,
    capture=None,
    check_distribution=None,
    predictions=True,
    atol=1e-4,
    rtol=1e-4,
):
    def pack(data, label):
        if data is None:
            return None
        if isinstance(data, list):
            return [pack(x, f"{label}-{i}") for i, x in enumerate(data)]
        path = folder / f"feed-{label}.bin"
        tensor = onnx.numpy_helper.from_array(data)
        if tensor.data_type == onnx.TensorProto.STRING:
            path.write_text(json.dumps([v.decode("utf8") for v in tensor.string_data]))
        else:
            path.write_bytes(tensor.raw_data)
        return {"file": str(path), "shape": list(data.shape), "type": tensor.data_type}

    inputs = {
        name: cache_names[name]
        if cache_names and name in cache_names
        else pack(data, i)
        for i, (name, data) in enumerate(feeds.items())
    }
    request = {"model": str(model), "directory": str(folder), "feeds": inputs}
    if external_data:
        request["externalData"] = external_data
    if isinstance(proc, DeviceRunner):
        reply = proc.run(request)
    else:
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError(f"Dart process exited with {proc.poll()}")
        reply = json.loads(line)
    if "error" in reply:
        raise RuntimeError(f"{reply['error']}\n{reply['trace']}")
    assert reply["outputs"].keys() == expected.keys()
    worst = 0.0

    def check(name, result, ref):
        nonlocal worst
        if ref is None:
            assert result is None
            return
        if isinstance(ref, list):
            assert isinstance(result, list) and len(ref) == len(result)
            for i, (r, v) in enumerate(zip(result, ref)):
                check(f"{name}[{i}]", r, v)
            return
        expected_type = onnx.numpy_helper.from_array(ref).data_type
        assert result["type"] == expected_type, (name, result["type"], expected_type)
        assert result["shape"] == list(ref.shape), (name, result["shape"], ref.shape)
        path = Path(result["file"])
        if expected_type == onnx.TensorProto.STRING:
            actual = np.array(json.loads(path.read_text()), dtype=object).reshape(
                ref.shape
            )
            ref = np.array(
                [
                    v.decode("utf8") if isinstance(v, bytes) else str(v)
                    for v in ref.flat
                ],
                dtype=object,
            ).reshape(ref.shape)
        else:
            tensor = onnx.TensorProto(
                data_type=expected_type, dims=ref.shape, raw_data=path.read_bytes()
            )
            actual = onnx.numpy_helper.to_array(tensor)
        if capture is not None and name in capture:
            capture[name] = actual
        if check_distribution is not None:
            check_distribution(name, actual)
            return
        if ref.dtype.kind in "biuO" or expected_type in [21, 22, 25, 26]:
            np.testing.assert_array_equal(actual, ref, err_msg=name)
        else:
            precision = "complex128" if ref.dtype.kind == "c" else "float64"
            actual = actual.astype(precision)
            ref = ref.astype(precision)
            np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol, err_msg=name)
        if predictions and (name == "logits" or len(expected) == 1):
            np.testing.assert_array_equal(actual.argmax(axis=-1), ref.argmax(axis=-1))
        if ref.dtype.kind in "fc":
            finite = np.isfinite(actual) & np.isfinite(ref)
            worst = max(
                worst, float(np.max(np.abs(actual[finite] - ref[finite]), initial=0))
            )

    for name, ref in expected.items():
        check(name, reply["outputs"][name], ref)
    return worst


def verify(name, model, binary, folder, browser=None, device=None):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    opts.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
    keys = [out.name for out in session.get_outputs()]
    if name.startswith("qwen"):
        tokens = np.array([[9707, 11, 1879]], dtype=np.int64)
        feeds = {
            "input_ids": tokens,
            "position_ids": np.arange(3, dtype=np.int64)[None, :],
            "attention_mask": np.ones((1, 3), dtype=np.int64),
        }
        for inp in session.get_inputs():
            if inp.name.startswith("past_key_values"):
                feeds[inp.name] = np.zeros(
                    (1, inp.shape[1], 0, inp.shape[3]), dtype=np.float32
                )
        cases = []
        for step in range(3):
            refs = dict(zip(keys, session.run(None, feeds)))
            sources = {
                k: k.replace("past_key_values", "present")
                for k in feeds
                if k.startswith("past_key_values")
            }
            cases.append((feeds, refs, sources if step else None))
            feeds = {k: refs[v] for k, v in sources.items()}
            feeds.update(
                input_ids=refs["logits"][:, -1, :].argmax(-1).astype(np.int64)[:, None],
                position_ids=np.array([[3 + step]], dtype=np.int64),
                attention_mask=np.ones((1, 4 + step), dtype=np.int64),
            )
    else:
        inp = session.get_inputs()[0]
        feeds = {
            inp.name: np.random.default_rng(2026)
            .normal(size=inp.shape)
            .astype(np.float32)
        }
        cases = [(feeds, dict(zip(keys, session.run(None, feeds))), None)]
    del session
    gc.collect()
    execution = (
        contextlib.closing(DeviceRunner(folder, binary, browser, device))
        if browser or device
        else subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
    )
    with execution as proc:
        try:
            for i, (feeds, expected, cache_names) in enumerate(cases):
                err = exchange(
                    proc,
                    folder,
                    model,
                    feeds,
                    expected,
                    cache_names,
                    atol=5e-4 if name.startswith("qwen") else 1e-4,
                )
                print(
                    f"{name} step {i + 1}: {len(expected)} output(s), max absolute error {err:.8g}",
                    flush=True,
                )
        finally:
            if not browser and not device:
                proc.stdin.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models", nargs="+", choices=[*MODELS, "qwen", "qwen_q4", "qwen3_q4"]
    )
    parser.add_argument(
        "--browser",
        metavar="CHROMIUM",
        help="Run inference in a headless browser and compare all outputs.",
    )
    parser.add_argument(
        "--device",
        help="Use the verification app on Android ('android') or an iOS simulator (UDID).",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--runner", type=Path, help="Previously compiled tool/compare_runner.dart"
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
        for name in args.models:
            verify(
                name,
                obtain(name, args.cache.resolve()),
                binary,
                folder,
                args.browser,
                args.device,
            )


if __name__ == "__main__":
    main()
