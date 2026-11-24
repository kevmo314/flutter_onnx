# Verification

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

The Qwen2.5 test prefills three prompt tokens into an empty cache, then runs two one-token decode steps. Each engine feeds its own 48 cache tensors back into its next step. The test asserts elementwise agreement on all 49 outputs per step (logits and every key/value tensor) and exact greedy-token agreement at every position. Relative tolerance is 0.0001. Absolute tolerance is 0.0005 for Qwen and 0.0001 for the vision models. All listed models were evaluated on the Dart VM CPU path. The same prefill and decode checks passed in Chromium for both Qwen2.5 exports and Qwen3. The Qwen2.5 four-bit export also passed on an Android emulator and an iPhone simulator. These full-model checks used the Dart CPU kernels. Qwen3 uses 56 cache tensors. The Qwen3.5 text test supplies the Dart decoder with the Dart embedding output and its own previous convolution, recurrent and key/value states. It checks all 49 decoder outputs at each step. The Qwen3.5 image-and-text run supplies four seeded input patches to the unmodified vision encoder. It compares its 1,024-element image feature, inserts each engine’s own feature in its input embeddings, and then checks prefill and two decode steps. Tokenization, image preprocessing, chat templates, long-context accuracy and throughput are outside these comparisons.

## Operator coverage

Run the official ONNX 1.23.2 standard-domain node fixtures through the Dart engine to audit opset 28. The harness counts unexpanded single-node cases, checks each staged model with the ONNX checker and compares outputs at the tolerances specified by the upstream case. It reports unrepresentable inputs, unavailable conversions and mismatches as failures. The strict suite covers **all 203 standard operators**. All 1,375 deterministic upstream cases pass at the specified tolerances. Three Bernoulli fixtures use random samples. The harness also checks those inputs across 8,192 draws per probability, verifies deterministic endpoints and accepts an absolute frequency error of 0.05. Fourteen operators have no qualifying upstream single-node fixture. Supplemental graphs exercise all fourteen on the same target and use 12,000 draws to check distribution moments, seeded replay, stream advancement and Dropout mask scaling.

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
python tool/verify_models.py --cache /tmp/onnx-models --browser chromium qwen_q4
```

`python tool/opset_conformance.py --strict` requires every catalogued operator to have test coverage and every case to pass. GitHub Actions runs the Dart VM, browser, WebGPU, Android emulator and iPhone simulator checks. Manually dispatch the Platform inference workflow to run the full four-bit Qwen comparison and strict opset suite on the Android emulator and iPhone simulator. The ONNX compatibility workflow runs the strict audit on the Dart VM and headless Chrome. Include `--browser chromium` when reproducing the audit in a local browser. The default push checks use the cached-attention and XOR fixtures. Physical mobile hardware remains a separate verification target.
