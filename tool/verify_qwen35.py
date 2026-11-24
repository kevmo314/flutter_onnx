"""Compare Qwen3.5-0.8B text embedding and hybrid decoder exports end to end."""

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path
import subprocess
import tempfile
import urllib.request

import numpy as np
import onnxruntime as ort
from verify_models import exchange

BASE = "https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX/resolve/8fe9dbe3fb39fe902e18df9be84b6a0ac77ae0f6/onnx/"
HASHES = {
    "embed_tokens_q4.onnx": "8773dcf4858f855bfce13def4356ca17e0bc516a477154496b6ffb6dd6d084dc",
    "embed_tokens_q4.onnx_data": "9210b1d26eb14136d3522584d17da14cc5f4b82b6f5607e6d80d59424a8cbe99",
    "decoder_model_merged_q4.onnx": "cfb338c5112e11e2a123d6d03d8870cd800883ee333ec0a25289482b609a16f8",
    "decoder_model_merged_q4.onnx_data": "fb38200610a2206aa4c10f4c5f41c08107e2b0b44968166943b3937a1eb11f84",
}


def run(cache, folder, exe, image=False):
    cache.mkdir(parents=True, exist_ok=True)
    files = dict(HASHES)
    if image:
        files.update(
            {
                "vision_encoder_q4.onnx": "bc3ecbf4d75759ccb3c197164777e07de9e0fdd606a07127e48b39385fdf6122",
                "vision_encoder_q4.onnx_data": "98aebedf02fc5414fd1c7f06a6580b42e272600ace9ecd33bfbc479a0c541c64",
            }
        )
    for name, expected in files.items():
        file = cache / name
        if not file.exists():
            tmp = file.with_suffix(".download")
            urllib.request.urlretrieve(BASE + name, tmp)
            tmp.replace(file)
        with file.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == expected, name
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.add_session_config_entry("session.disable_prepacking", "1")
    embed_path = cache / "embed_tokens_q4.onnx"
    decode_path = cache / "decoder_model_merged_q4.onnx"

    def oracle(path):
        return ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    embed, decode = oracle(embed_path), oracle(decode_path)
    key = embed.get_outputs()[0].name
    names = [o.name for o in decode.get_outputs()]
    state = {
        inp.name: np.zeros(
            tuple(
                1 if d == "batch_size" else 0 if d == "past_sequence_length" else d
                for d in inp.shape
            ),
            np.float32,
        )
        for inp in decode.get_inputs()
        if inp.name.startswith("past")
    }
    parents = {
        k: k.replace("past_key_values", "present").replace("past_", "present_")
        for k in state
    }
    tokens = np.array([[9707, 11, 1879]], np.int64)
    pos = 0

    def side(path):
        return {path.name + "_data": str(path) + "_data"}

    with ExitStack() as stack:

        def process():
            proc = stack.enter_context(
                subprocess.Popen(
                    [str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
                )
            )
            stack.callback(proc.stdin.close)
            return proc

        embedding, decoder = process(), process()
        if image:
            path = cache / "vision_encoder_q4.onnx"
            vision_inputs = {
                "pixel_values": np.random.default_rng(314)
                .normal(size=(4, 1536))
                .astype(np.float32),
                "image_grid_thw": np.array([[1, 2, 2]], np.int64),
            }
            reference_features = oracle(path).run(None, vision_inputs)[0]
            features = {"image_features": None}
            diff = exchange(
                process(),
                folder,
                path,
                vision_inputs,
                {"image_features": reference_features},
                capture=features,
                external_data=side(path),
                predictions=False,
                atol=5e-4,
            )
            print(
                f"Vision encoder: {reference_features.shape}, maximum absolute error {diff:.8g}",
                flush=True,
            )
            # One spatially merged image token has identical three-axis MRoPE positions.
            tokens = np.array([[248053, 248056, 248054, 9707]], np.int64)
        for step in range(3):
            feed = {"input_ids": tokens}
            vector = embed.run(None, feed)[0]
            observed = {key: None}
            exchange(
                embedding,
                folder,
                embed_path,
                feed,
                {key: vector},
                capture=observed,
                external_data=side(embed_path),
                predictions=False,
            )
            if image and step == 0:
                vector[:, 1] = reference_features
                observed[key] = observed[key].copy()
                observed[key][:, 1] = features["image_features"]
            count = tokens.shape[1]
            inputs = {
                "inputs_embeds": vector,
                "position_ids": np.tile(
                    np.arange(pos, pos + count, dtype=np.int64)[None, None, :],
                    (3, 1, 1),
                ),
                "attention_mask": np.ones((1, pos + count), np.int64),
                **state,
            }
            ref = dict(zip(names, decode.run(None, inputs), strict=True))
            inputs["inputs_embeds"] = observed[key]
            diff = exchange(
                decoder,
                folder,
                decode_path,
                inputs,
                ref,
                parents if step else None,
                external_data=side(decode_path),
                atol=5e-4,
                rtol=1e-4,
            )
            print(
                f"Qwen3.5 {'image and text' if image else 'text'} step {step + 1}: embedding + {len(names)} decoder outputs, maximum absolute error {diff:.8g}",
                flush=True,
            )
            tokens = ref["logits"][:, -1, :].argmax(-1).astype(np.int64)[:, None]
            state = {k: ref[v] for k, v in parents.items()}
            pos += count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--runner", type=Path)
    p.add_argument(
        "--image",
        action="store_true",
        help="Include the vision encoder and feed its features to the hybrid decoder.",
    )
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        exe = args.runner.resolve() if args.runner else folder / "compare"
        if not args.runner:
            subprocess.run(
                ["dart", "compile", "exe", "tool/compare_runner.dart", "-o", str(exe)],
                check=True,
            )
        run(args.cache.resolve(), folder, exe, args.image)


if __name__ == "__main__":
    main()
