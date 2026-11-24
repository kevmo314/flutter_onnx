# Verification

The strict opset 28 suite passed on the Dart VM, Chromium, an Android API 35 emulator and an iPhone simulator. The platform tests recorded a WebGPU dispatch on Chromium’s software Vulkan adapter and a Metal Performance Shaders dispatch on the iPhone simulator. The Android emulator used the Dart CPU fallback. Physical-device acceleration and throughput remain unmeasured. [Full-model and browser results](https://github.com/kevmo314/flutter_onnx/actions/runs/45369324401) · [Platform results and per-device reports](https://github.com/kevmo314/flutter_onnx/actions/runs/45369319905)

## Full-model comparisons

The independent oracle is Python ONNX Runtime 1.32.0 on CPU. The comparison runs unmodified exports with pinned revisions and verifies every output's name, element type, dimensions and values. Each classifier also checks the predicted class on a seeded random 224 × 224 RGB input. The test downloads model weights into a local cache and verifies their SHA-256 hashes. The example bundles small XOR and cached-attention fixtures.

| Model | Export | Maximum absolute difference |
| --- | --- | ---: |
| SqueezeNet 1.1 | ONNX Model Zoo, opset 17 | 0.00000287 |
| MobileNetV2 0.50 | ONNX Model Zoo, opset 17 | 0.00000334 |
| ResNet-18 | ONNX Model Zoo, opset 17 | 0.00000144 |
| EfficientNet-B0 | ONNX Model Zoo, opset 17 | 0.00000108 |
| DenseNet-121 | ONNX Model Zoo, opset 17 | 0.00000334 |
| MobileNetV3 Small 0.50 | ONNX Model Zoo, opset 17 | 0.00000477 |
| Qwen2.5-0.5B-Instruct | FP32, opset 14 | 0.000182 |
| Qwen2.5-0.5B-Instruct | Four-bit block weights, opset 14 | 0.0000482 |
| Qwen3-0.6B | Four-bit block weights, opset 14 | 0.000153 |
| Qwen3.5-0.8B | Four-bit text embedding and hybrid decoder, opset 21 | 0.0000405 |
| Qwen3.5-0.8B | Four-bit vision encoder, text embedding and hybrid decoder, opset 21 | 0.0000531 |

| Execution target | Verified model coverage |
| --- | --- |
| Dart VM and Chromium | All exports in the table, including text-only and image-plus-text Qwen3.5 |
| Android API 35 emulator and iPhone simulator | Qwen2.5-0.5B-Instruct four-bit and Qwen3-0.6B four-bit |

Full-model checks use the Dart CPU kernels. Each Qwen comparison exercises a multi-token prefill and two cached one-token decode steps. At every step it checks the complete logits, every state tensor, and exact greedy token IDs at every position. Relative tolerance is 0.0001. Absolute tolerance is 0.0005 for Qwen and 0.0001 for classifiers.

Qwen2.5 takes INT64 `input_ids`, `position_ids` and `attention_mask` plus empty FLOAT caches shaped `[batch, kv_heads, 0, head_dim]`. Each engine passes its own `present.*` outputs back as `past_key_values.*` and extends the mask for each token. Qwen2.5 returns 48 cache tensors and Qwen3 returns 56.

Qwen3.5 uses each engine’s own token embeddings, convolution and recurrent states, and key/value caches. The image-and-text comparison also runs the vision encoder on four seeded image patches, checks the 1,024-element feature, inserts it in the prompt embeddings and checks all 49 decoder outputs at each step. These comparisons begin with token IDs and preprocessed patches. Tokenization, image preprocessing, chat templates, long-context accuracy and throughput are outside their scope.

## Operator coverage

Run the official ONNX 1.23.2 standard-domain node fixtures through the Dart engine to audit opset 28. The harness counts unexpanded single-node cases, checks each staged model with the ONNX checker and compares outputs at the tolerances specified by the upstream case. It reports unrepresentable inputs, unavailable conversions and mismatches as failures. The strict suite covers **all 203 standard operators**. All 1,375 deterministic upstream cases pass at the specified tolerances. Three Bernoulli fixtures use random samples. The harness also checks those inputs across 8,192 draws per probability, verifies deterministic endpoints and accepts an absolute frequency error of 0.05. Fourteen operators have no qualifying upstream single-node fixture. Supplemental graphs exercise all fourteen on the same target and use 12,000 draws to check distribution moments, seeded replay, stream advancement and Dropout mask scaling.

ONNX generates its Fourier reference values with the host NumPy FFT library. On the 10-point inverse fixture, the same pocketfft source produced a near-zero imaginary component of 0 with fused operations and 1.91 × 10⁻⁷ with separate operations. That difference exceeds the fixture’s 10⁻⁷ absolute tolerance. The suite keeps all upstream DFT inputs and outputs in one versioned fixture so the browser, Android and iOS receive identical bytes and tolerances. Reproduce it with ONNX 1.23.2 and NumPy 2.3.5 on x86-64 Linux using `python tool/freeze_dft.py`. A second check compares forward and inverse transforms of seeded complex signals at four lengths against double-precision NumPy results on each host.

The Scan-8 fixture is converted to opset 28 with the official converter and exercised once per original batch item. For MeanVarianceNormalization the harness runs shape inference with explicit schema defaults to work around a checker expansion issue, then executes the original graph. The coverage table describes the combinations exercised by this suite. Other shapes, attributes, data types and exporter-specific extensions can require additional verification.

[Per-operator results](tool/conformance.tsv) · [Passing-case regression baseline](tool/conformance-baseline.txt)

## Reproduce

Use Python 3.12 with `onnx==1.23.2`, `onnxruntime==1.32.0` and NumPy. Run from the repo root.

```sh
dart pub get
dart test -p vm -p chrome
python tool/verify_randomized.py
python tool/opset_conformance.py --strict --baseline tool/conformance-baseline.txt --report report.json
python tool/verify_models.py --cache /tmp/onnx-models squeezenet1_1 mobilenetv2_050 resnet18 efficientnet_b0 densenet121 mobilenetv3_small_050
python tool/verify_models.py --cache /tmp/onnx-models qwen qwen_q4 qwen3_q4
python tool/verify_qwen35.py --cache /tmp/onnx-models/qwen35
python tool/verify_qwen35.py --cache /tmp/onnx-models/qwen35 --image
python tool/verify_qwen35.py --cache /tmp/onnx-models/qwen35 --image --browser chromium
python tool/verify_models.py --cache /tmp/onnx-models --browser chromium qwen_q4
```

Pushes and pull requests run the compatibility, platform inference, and RTChat mobile speech workflows. ONNX compatibility runs randomized differential checks on the Dart VM, then compares the full opset suite, all six classifiers, and the Qwen text and vision models on the VM and in Chromium. Platform inference runs analysis, unit tests, the Flutter web build, a required WebGPU dispatch, and mobile integration tests. Both mobile targets run four-bit Qwen2.5 and Qwen3 and the strict opset suite. Each target uploads its operator report to the workflow run. Physical mobile hardware remains a separate verification target.

### RTChat playback workload

RTChat mobile speech CI builds the migrated app against the runtime revision under test. It streams one English chat message in three phrases with Supertonic 3, voice M2, speed 1 and five denoising steps on an Android x86-64 emulator (Flutter profile build) and an iPhone simulator (Flutter debug build). CI builds the iOS app on a separate runner so Xcode compilation and indexing do not compete with playback on the measurement host. The report records model load time and native first-audio and playback-complete events for a cold utterance and two warm utterances.

Every streamed run must start playback before inference produces the final PCM chunk. One additional warm run synthesizes the same words as a single segment to measure its startup cost. After playback, the app consumes the public PCM stream in three rounds with different noise seeds. It records the time spent in the public synthesis API and divides it by the duration of the returned PCM. Every round must have a real-time factor below 1. The artifacts also include per-segment real-time factors. Playback completion includes the time the audio device spends playing the queued samples. The first-audio event measures player-side startup. Acoustic latency requires a physical output measurement.

The native CPU job also runs the same three segments through Dart AOT on x86-64 Linux, Arm64 Linux and Apple Silicon macOS. It checks the full waveform and predicted duration against Supertone, requires exact trimmed lengths, and enforces faster-than-real-time synthesis in both warm rounds. Its report breaks down time spent in duration prediction, text encoding, denoising and the vocoder. It estimates queue drain and starvation under overlapping synthesis and playback. Separate multilingual tests cover five utterances each for Supertonic 2 and 3 and enforce the same exact trimmed lengths.
