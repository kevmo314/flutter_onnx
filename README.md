# flutter_onnx

Load an ONNX graph once and run it locally on Flutter web, Android, or iOS. The Dart package implements protobuf model decoding, graph execution, tensor broadcasting and CPU kernels. ImageDecoder uses Dart image codecs, including JPEG 2000. A companion Flutter plugin can send matrix products to the device.

## Install

For CPU and device execution, add the companion package to your application's `pubspec.yaml`. It resolves and exports the core API from the same Git revision.

```yaml
dependencies:
  flutter_onnx_accelerators:
    git:
      url: https://github.com/kevmo314/flutter_onnx.git
      ref: main
      path: accelerators
flutter:
  assets:
    - assets/xor.onnx
```

Copy the included model from [example/assets/xor.onnx](example/assets/xor.onnx) or supply your own model. The example graph takes two bits and returns their exclusive OR.

```dart
import 'package:flutter/services.dart';
import 'package:flutter_onnx_accelerators/flutter_onnx_accelerators.dart';

Future<void> predict() async {
  final asset = await rootBundle.load('assets/xor.onnx');
  final session = OnnxSession.fromBytes(
    asset.buffer.asUint8List(asset.offsetInBytes, asset.lengthInBytes),
  );
  final inputs = {'bits': Tensor.float32([1, 2], [1, 0])};
  final backend = await createMatMulBackend();
  try {
    final output = await session.runAsync(inputs, backend: backend);
    print(output['xor']!.values.single); // 1.0
  } finally {
    await backend?.close();
  }
}
```

For CPU-only use, install the core package with `flutter pub add flutter_onnx --git-url https://github.com/kevmo314/flutter_onnx.git`, import `package:flutter_onnx/flutter_onnx.dart` and call `session.run(inputs)`. Reuse the session and backend across predictions. Inspect `session.inputs` and `session.outputs` for names, element types and declared dimensions. Named symbolic dimensions bind independently on each call.

## Device kernels

| Target | Backend | Dispatch |
| --- | --- | --- |
| Web | WebGPU | A WGSL compute shader multiplies two FLOAT matrices. Requires WebGPU in a secure context such as HTTPS or localhost. |
| Android | Android Neural Networks API | On API 29+ a supported GPU or accelerator executes a fully connected operation. Other Android devices use Dart. NNAPI was deprecated in Android 15 and availability varies by driver. |
| iOS | Metal Performance Shaders | A Metal command queue executes MPSMatrixMultiplication on compatible devices. The plugin targets iOS 13+. |

`runAsync` offloads nonempty FLOAT `MatMul` with shared rank-two weights and the matrix product in `Gemm`. Transposition, scaling, bias and all other operators run in Dart. When a device declines a product, the scheduler computes it on the CPU. Pass `onAccelerated: (node) => print(node)` to observe successful dispatches. Transfers, weight transposition and compilation can outweigh device computation for small or frequently repeated products. The implementations currently transfer buffers per call.

## Model compatibility

The compatibility target is the standard ONNX domain at **opset 28** (ONNX 1.23.2). All 203 standard-domain operators pass the strict verification suite, including 1,375 deterministic upstream cases, three Bernoulli distribution checks and supplemental graphs for the 14 operators without upstream single-node fixtures. The loader accepts binary graphs declaring opsets 13–28 and IR versions 3–14. A supported version number alone does not establish that all operators, types and attributes in a particular graph will execute.

The [verification report](VERIFICATION.md) records upstream node-test coverage and actual model results. The runtime includes NCHW convolution, grouped convolution, pooling, matrix products, broadcasting, indexing, reductions, quantization, normalization, rotary embeddings and grouped-query attention with caches. The tested Qwen2.5, Qwen3 and Qwen3.5 exports use additional kernels for four-bit block weights, quantized embeddings and fused transformer operations. Initializers can use inline bytes, protobuf values, sparse coordinates or external weight files. Pass external files to `OnnxSession.fromBytes(modelBytes, externalData: {'weights.bin': weightBytes})` using the locations named in the model. Use `runValues` for graphs with sequences and optional values. Sequence elements can have different shapes, including across loop iterations. Model-local functions can call standard kernels and bind attributes at their call sites. Graph inputs and outputs support dense tensors, sequences and optionals. Map-valued graph interfaces and segmented TensorProto storage are outside the current loader’s coverage.

CPU inference is synchronous. Mobile devices and browsers impose different memory limits. Six classifiers and the tested Qwen2.5, Qwen3 and Qwen3.5 pipelines passed on the Dart VM and in Chromium. Both four-bit Qwen2.5 and Qwen3 passed on an Android emulator and an iPhone simulator. Mobile integration tests check a bundled cached-attention graph and all four XOR inputs. The WebGPU test requires a real compute dispatch on a software adapter.

## Run the example and tests

```sh
cd example
flutter pub get
flutter run -d chrome
# Or use flutter devices and pass an Android or iOS device ID to flutter run -d.
```

From the repository root, run `dart pub get` and `dart test -p vm -p chrome`. The tests compare graph outputs with committed vectors generated by ONNX's Python reference evaluator. The Python tools also run independent full-model and operator comparisons. From `example`, run `flutter test integration_test/inference_test.dart -d <device-id>` to exercise the bundled model on mobile. The dedicated browser test can require a successful WebGPU dispatch with `flutter test --platform chrome --dart-define=REQUIRE_WEBGPU=true test/webgpu_test.dart`.

The JPEG reconstruction code is based in part on the work of the Independent JPEG Group. The WebP decoder adapts the Dart image package and libwebp. [Independent JPEG Group terms](third_party/ijg/README.ijg) · [Dart image license](third_party/image/LICENSE) · [libwebp license](third_party/libwebp/COPYING)

The optional [native CPU package](cpu/) supplies `NativeCpuBackend` for
`runAsync`. Reuse the backend and session in an inference isolate so that
pointwise convolutions can reuse their prepared filters. The backend uses
Accelerate on Apple devices and SIMD kernels on Android and Linux. It also
handles depthwise convolution and normalization across channels in NCL layout.

The RTChat mobile speech workflow builds the app with the runtime checkout
and exercises its native audio player. It measures three fresh synthesis
rounds with the default five-step Supertonic 3 model. Each round must generate
its audio in less time than the PCM takes to play. The JSONL artifact records
synthesis time, audio duration, real-time factor, time to first audio and
playback completion. The platform inference workflow checks the waveforms
against the Supertone Python implementation on Linux and macOS.
