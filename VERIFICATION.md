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

Every streamed run must start playback before inference produces the final PCM chunk. One additional warm run synthesizes the same words as a single segment. Mean warm startup with streaming must beat that baseline on the same player. The job also reports how long synthesis continued after first audio and how much playback elapsed time exceeded the PCM duration. After playback, the app consumes the public PCM stream in three rounds with different noise seeds. It records the time spent in the public synthesis API and divides it by the duration of the returned PCM. Every round must have a real-time factor below 1. The artifacts also include per-segment real-time factors. Playback completion includes the time the audio device spends playing the queued samples. The first-audio event measures player-side startup. Acoustic latency requires a physical output measurement.

The native CPU job also runs the same three segments through Dart AOT on x86-64 Linux, Arm64 Linux and Apple Silicon macOS. It checks the full waveform and predicted duration against Supertone, requires exact trimmed lengths, and enforces faster-than-real-time synthesis in both warm rounds. Its report breaks down time spent in duration prediction, text encoding, denoising and the vocoder. It estimates queue drain and starvation under overlapping synthesis and playback. Separate multilingual tests cover five utterances each for Supertonic 2 and 3 and enforce the same exact trimmed lengths.

With RTChat pinned to runtime revision `7abf879`, the Android emulator generated 6.749 seconds of audio in 2.15–2.25 seconds. Warm playback started in 0.50–0.52 seconds when streaming, compared with 1.52 seconds when speaking the same words as one segment. On the iPhone simulator, synthesis took 5.26–5.99 seconds. Warm streamed startup took 1.19–1.73 seconds, compared with 4.13 seconds for the whole message. Both platforms also passed all nine mobile integration tests. Each of the five multilingual waveform comparisons matched the reference sample count with peak absolute error below 0.001.

[Android migration and timing](https://github.com/kevmo314/flutter_onnx/actions/runs/45524923783/job/138905776400) · [iOS migration and timing](https://github.com/kevmo314/flutter_onnx/actions/runs/45524923783/job/138905776012)

Across five iOS runs of that revision, 13 of 15 synthesis rounds beat real time. The two misses took 6.785 and 6.909 seconds. Even with faster-than-real-time synthesis for the utterance, playback can run out of queued samples when a later phrase takes longer to generate. The pinned iOS run took 1.18–1.47 seconds longer to play the warm utterance than its PCM duration, measured from the first-audio event. The raw reports include model loading and cold startup separately.

[First iOS integration](https://github.com/kevmo314/flutter_onnx/actions/runs/45519231462/job/138886546550) · [Second iOS profile](https://github.com/kevmo314/flutter_onnx/actions/runs/45519744684/job/138888304617) · [160 ms miss](https://github.com/kevmo314/flutter_onnx/actions/runs/45522054055/job/138899807720) · [36 ms miss](https://github.com/kevmo314/flutter_onnx/actions/runs/45523781530/job/138907028628)

A same-host Apple Silicon AOT comparison ran the three-phrase benchmark with native projection pairing disabled, enabled, enabled and disabled, using one binary and the same inputs. Mean synthesis time across the four warm rounds per configuration fell from 3.776 to 3.068 seconds (18.7%). Each variant compared the waveform with Supertone. [Paired timings](https://github.com/kevmo314/flutter_onnx/actions/runs/45527657241/job/138914874115)

With projection pairing enabled in RTChat's pinned runtime (`c48e99e`), the Android integration job produced 6.749 seconds of audio in 3.13–3.22 seconds. Warm player startup was 0.81–0.83 seconds, compared with 2.22 seconds for the whole message. The independent iPhone simulator timing job produced the same amount of audio in 4.19–5.74 seconds. Warm player startup was 1.37–1.59 seconds, compared with 4.80 seconds for the whole message. On that iOS run, cold first audio took 9.31 seconds. From warm first audio to playback completion, elapsed time exceeded the PCM duration by 1.49–2.07 seconds. This includes any queue starvation and player overhead.

[Android integration and timings](https://github.com/kevmo314/flutter_onnx/actions/runs/45528460272/job/138917536653) · [iOS timing](https://github.com/kevmo314/flutter_onnx/actions/runs/45527632320/job/138920393163) · [App, plugin, Kotlin and Swift suites](https://github.com/kevmo314/flutter_onnx/actions/runs/45528460280)

Four independent iPhone simulator jobs tested the paired projection code through RTChat. Ten of their twelve seeded synthesis rounds finished faster than real time. The two misses took 7.652 and 6.955 seconds for 6.749 seconds of audio. Both jobs correctly failed their timing gate. On the same four runners, warm first audio took 1.21–1.93 seconds while streaming and 4.80–5.74 seconds when synthesizing the whole message. Cold first audio after model loading took 3.43–9.31 seconds. The migration pinned to `c48e99e` passed all nine integration tests on each platform, including five reference waveforms per platform with exact sample counts and peak absolute error below 0.001. These runs establish earlier playback with streaming. Consistent faster-than-real-time iPhone performance still needs physical-device verification.

[Independent timing run](https://github.com/kevmo314/flutter_onnx/actions/runs/45527632320/job/138920393163) · [Pinned migration](https://github.com/kevmo314/flutter_onnx/actions/runs/45528460272/job/138917536366) · [Second integration runner](https://github.com/kevmo314/flutter_onnx/actions/runs/45527282756/job/138913653702) · [Published runtime timing gate](https://github.com/kevmo314/flutter_onnx/actions/runs/45530561297/job/138929147972)
