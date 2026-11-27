"""Measure the previous RTChat CPU provider against the compiled Dart playback driver."""
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def measure(root):
    manifest = json.loads((root / "manifest.json").read_text())
    model = Path(manifest["model_dir"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    graphs = [
        ort.InferenceSession(str(model / "onnx" / (name + ".onnx")), options,
                             providers=["CPUExecutionProvider"])
        for name in ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
    ]
    rows = []
    for repetition in range(6):
        wall = cpu = first = 0
        for case in manifest["cases"]:
            def fixture(name):
                entry = case["tensors"][name]
                return np.fromfile(root / entry["file"], dtype=entry["dtype"]).reshape(entry["shape"])

            styles = json.loads((model / "voice_styles" / (case["voice"] + ".json")).read_text())

            def style(name):
                return np.asarray(styles[name]["data"], dtype=np.float32).reshape(styles[name]["dims"])

            ids, latent = fixture("text_ids"), fixture("noise")
            text_mask = np.ones((1, 1, ids.shape[-1]), dtype=np.float32)
            latent_mask = np.ones((1, 1, latent.shape[-1]), dtype=np.float32)
            duration_style, context = style("style_dp"), style("style_ttl")
            total = np.array([case["steps"]], dtype=np.float32)
            cpu_start, start = time.process_time_ns(), time.perf_counter_ns()
            duration, = graphs[0].run(None, dict(text_ids=ids, text_mask=text_mask, style_dp=duration_style))
            text, = graphs[1].run(None, dict(text_ids=ids, text_mask=text_mask, style_ttl=context))
            for step in range(case["steps"]):
                latent, = graphs[2].run(None, dict(noisy_latent=latent, text_emb=text, text_mask=text_mask,
                                                   latent_mask=latent_mask, style_ttl=context,
                                                   current_step=np.array([step], dtype=np.float32), total_step=total))
            wave, = graphs[3].run(None, {"latent": latent})
            wall += (time.perf_counter_ns() - start) / 1000
            cpu += (time.process_time_ns() - cpu_start) / 1000
            if not first:
                first = wall
            expected = np.fromfile(root / case["pcm"], dtype="<f4")
            samples = int(duration.item() / np.float32(case["speed"]) * np.float32(case["sample_rate"]))
            if samples != expected.size or not np.allclose(wave.reshape(-1)[:samples], expected,
                                                           rtol=0, atol=manifest["tolerances"]["wav"]):
                raise ValueError(f"Reference mismatch for {case['name']}")
        rows.append(dict(round=repetition + 1, synthesis_us=wall, cpu_us=cpu, first_pcm_us=first))
    return rows


if __name__ == "__main__":
    root, log, output = map(Path, sys.argv[1:])
    dart = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    dart = [row for row in dart if "synthesis_us" in row]
    reference = measure(root)
    report = {"onnxruntime_version": ort.__version__, "flutter_onnx": dart, "onnxruntime": reference}
    if len(dart) != len(reference):
        raise ValueError("Incomplete Dart timing samples")
    label = "Metal and native CPU" if "metal_dispatches" in dart[-1] else "Native CPU"
    summary = [
        f"### Supertonic 3 on {label}",
        "",
        "Warm medians for the same 6.75 seconds of speech (five messages per engine).",
        "",
        "| Metric | flutter_onnx | ONNX Runtime CPU |",
        "| --- | ---: | ---: |",
    ]
    for metric in ("synthesis_us", "cpu_us", "first_pcm_us"):
        medians = [statistics.median(sample[metric] for sample in rows[1:]) for rows in (dart, reference)]
        if min(medians) <= 0:
            raise ValueError(f"Invalid {metric}: {medians}")
        print(f"{metric}: flutter_onnx={medians[0]:.0f} onnxruntime={medians[1]:.0f} ratio={medians[0]/medians[1]:.3f}")
        title = {"synthesis_us": "Synthesis", "cpu_us": "Process CPU", "first_pcm_us": "First PCM"}[metric]
        summary.append(f"| {title} | {medians[0] / 1e6:.3f} s | {medians[1] / 1e6:.3f} s |")
    output.write_text(json.dumps(report, indent=2))
    if "GITHUB_STEP_SUMMARY" in os.environ:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as step_summary:
            print("\n".join(summary), file=step_summary)
