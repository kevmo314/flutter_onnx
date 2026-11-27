import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:flutter_onnx_cpu/flutter_onnx_cpu.dart';
import 'package:test/test.dart';

class _IndividualProjectionBackend extends NativeCpuBackend {
  @override
  bool get supportsPointwisePairs => false;
}

class _DecliningPairBackend extends NativeCpuBackend {
  var calls = 0;
  @override
  bool get supportsPointwisePairs => true;
  @override
  Future<Float32List?> pointwisePair(
    Tensor input,
    Tensor firstWeights,
    Tensor? firstBias,
    Tensor secondWeights,
    Tensor? secondBias,
  ) async {
    calls++;
    return null;
  }
}

class _DecliningNormalizedBackend extends _DecliningPairBackend
    implements NormalizedProjectionBackend {
  var normalizedCalls = 0;

  @override
  Future<Float32List?> normalizedPair(
    Tensor input,
    Tensor scale,
    Tensor normBias,
    double epsilon,
    Tensor firstWeights,
    Tensor? firstBias,
    Tensor secondWeights,
    Tensor? secondBias,
  ) async {
    normalizedCalls++;
    return null;
  }
}

void main() {
  const backend = NativeCpuBackend();
  test('matrix row and column tails include byte offsets', () async {
    const m = 5;
    for (final k in [11, 191, 192, 193, 385]) {
      for (final n in [1, 15, 16, 17, 33]) {
        final a = Float32List.fromList(
          List.generate(m * k, (i) => (i % 5 - 2).toDouble()),
        );
        final raw = Float32List.fromList(
          List.generate(k * n + 7, (i) => (i % 7 - 3).toDouble()),
        );
        final b = Float32List.sublistView(raw, 7);
        final out = await backend.multiply(a, b, rows: m, inner: k, columns: n);
        expect(out, [
          for (var i = 0; i < m; i++)
            for (var j = 0; j < n; j++)
              List.generate(
                k,
                (p) => a[i * k + p] * b[p * n + j],
              ).reduce((a, b) => a + b),
        ]);
      }
    }
  });
  test(
    'parallel row ranges match a serial product with irregular dimensions',
    () async {
      const m = 321, k = 257, n = 257;
      final a = Float32List.fromList(
        List.generate(m * k, (i) => (i % 7 - 3).toDouble()),
      );
      final b = Float32List.fromList(
        List.generate(k * n, (i) => (i % 11 - 5).toDouble()),
      );
      final serial = await const NativeCpuBackend(
        maxThreads: 1,
      ).multiply(a, b, rows: m, inner: k, columns: n);
      final parallel = await backend.multiply(
        a,
        b,
        rows: m,
        inner: k,
        columns: n,
      );
      expect(parallel, orderedEquals(serial!));
    },
  );

  test('matrix operands broadcast across independent batch axes', () async {
    final graph = OnnxSession.fromBytes(
      base64Decode(
        'CA46cQoRCgFhCgFiEgF5IgZNYXRNdWwSBWJhdGNoWhsKAWESFgoUCAESEAoCCAIKAggBCgIIAQoCCANaGwoBYhIWChQIARIQCgIIAQoCCAMKAggDCgIIAWIbCgF5EhYKFAgBEhAKAggCCgIIAwoCCAEKAggBQgQKABAS',
      ),
    );
    final feeds = {
      'a': Tensor.float32([2, 1, 1, 3], [1, 2, 3, 4, 5, 6]),
      'b': Tensor.float32([1, 3, 3, 1], [1, 0, 0, 0, 1, 0, 0, 0, 1]),
    };
    final nodes = <String>[];
    final result = await graph.runAsync(
      feeds,
      backend: backend,
      onAccelerated: nodes.add,
    );
    expect(result['y']!.shape, [2, 3, 1, 1]);
    expect(result['y']!.values, [1, 2, 3, 4, 5, 6]);
    expect(nodes, hasLength(1));
  });

  test('batched pointwise convolution preserves NCL order', () async {
    final x = Tensor.float32([2, 3, 5], List.generate(30, (i) => i - 11));
    final w = Tensor.float32([4, 3, 1], List.generate(12, (i) => i % 4 - 2));
    final bias = Tensor.float32([4], [4, 2, -1, 3]);
    for (final hasBias in [false, true]) {
      final graph = OnnxSession.fromBytes(
        base64Decode(
          hasBias
              ? 'CA46gQEKFQoBeAoBdwoEYmlhcxIBeSIEQ29udhIJcG9pbnR3aXNlWhcKAXgSEgoQCAESDAoCCAIKAggDCgIIBVoXCgF3EhIKEAgBEgwKAggECgIIAwoCCAFaEgoEYmlhcxIKCggIARIECgIIBGIXCgF5EhIKEAgBEgwKAggCCgIIBAoCCAVCBAoAEBI='
              : 'CA46ZwoPCgF4CgF3EgF5IgRDb252Eglwb2ludHdpc2VaFwoBeBISChAIARIMCgIIAgoCCAMKAggFWhcKAXcSEgoQCAESDAoCCAQKAggDCgIIAWIXCgF5EhIKEAgBEgwKAggCCgIIBAoCCAVCBAoAEBI=',
        ),
      );
      final feeds = {'x': x, 'w': w, if (hasBias) 'bias': bias};
      final accelerated = <String>[];
      final output = (await graph.runAsync(
        feeds,
        backend: backend,
        onAccelerated: accelerated.add,
      ))['y']!;
      expect(output.shape, [2, 4, 5]);
      expect(output.values, graph.run(feeds)['y']!.values);
      expect(accelerated, hasLength(1));
    }
  });

  test(
    'prepared convolution spans batches, tails and changed biases',
    () async {
      const width = 13, batch = 2;
      for (final (channels, filters) in [(96, 128), (256, 128), (128, 1024)]) {
        final w = Tensor.float32([
          filters,
          channels,
          1,
        ], List.generate(filters * channels, (i) => (i % 7) - 3));
        for (final shift in [0, 1]) {
          final x = Tensor.float32(
            [batch, channels, width],
            List.generate(
              batch * channels * width,
              (i) => ((i + shift) % 11) - 5,
            ),
          );
          final bias = shift == 0
              ? null
              : Tensor.float32([filters], List.generate(filters, (i) => i % 5));
          final out = await backend.pointwise(x, w, bias, gelu: false);
          expect(out, isNotNull);
          for (var b = 0; b < batch; b++) {
            for (var f = 0; f < filters; f++) {
              for (var t = 0; t < width; t++) {
                num sum = bias?.values[f] ?? 0;
                for (var c = 0; c < channels; c++) {
                  sum +=
                      w.values[f * channels + c] *
                      x.values[(b * channels + c) * width + t];
                }
                expect(out![(b * filters + f) * width + t], sum);
              }
            }
          }
        }
      }
    },
  );
  test(
    'declined projection pairs preserve outputs and visible activations',
    () async {
      final inputs = {
        'x': Tensor.float32([1, 2, 3], [1, 0, 2, -1, 0, -2]),
      };
      for (final (encoded, exposed) in [
        (
          'CAk6kAMKGAoBeAoCdzESAWUaBmV4cGFuZCIEQ29udgoTCgFlCgFkEgJkMBoCZDAiA0RpdgoTCgJkMBIDZXJmGgNlcmYiA0VyZgobCgNlcmYKA29uZRIEcGx1cxoEcGx1cyIDQWRkCiAKAWUKBHBsdXMSB3Byb2R1Y3QaB3Byb2R1Y3QiA011bAoiCgdwcm9kdWN0CgRoYWxmEgZoaWRkZW4aBGdlbHUiA011bAohCgZoaWRkZW4KAncyEgNvdXQaCGNvbnRyYWN0IgRDb252EgRwYWlyKi4IBAgCCAEQAUICdzFKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKi4IAggECAEQAUICdzJKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKg4QAUIEaGFsZkoEAAAAPyoNEAFCA29uZUoEAACAPyoLEAFCAWRKBPMEtT9aFwoBeBISChAIARIMCgIIAQoCCAIKAggDYhkKA291dBISChAIARIMCgIIAQoCCAIKAggDQgQKABAS',
          false,
        ),
        (
          'CAk6rgMKGAoBeAoCdzESAWUaBmV4cGFuZCIEQ29udgoTCgFlCgFkEgJkMBoCZDAiA0RpdgoTCgJkMBIDZXJmGgNlcmYiA0VyZgobCgNlcmYKA29uZRIEcGx1cxoEcGx1cyIDQWRkCiAKAWUKBHBsdXMSB3Byb2R1Y3QaB3Byb2R1Y3QiA011bAoiCgdwcm9kdWN0CgRoYWxmEgZoaWRkZW4aBGdlbHUiA011bAohCgZoaWRkZW4KAncyEgNvdXQaCGNvbnRyYWN0IgRDb252EgRwYWlyKi4IBAgCCAEQAUICdzFKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKi4IAggECAEQAUICdzJKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKg4QAUIEaGFsZkoEAAAAPyoNEAFCA29uZUoEAACAPyoLEAFCAWRKBPMEtT9aFwoBeBISChAIARIMCgIIAQoCCAIKAggDYhkKA291dBISChAIARIMCgIIAQoCCAIKAggDYhwKBmhpZGRlbhISChAIARIMCgIIAQoCCAQKAggDQgQKABAS',
          true,
        ),
      ]) {
        final graph = OnnxSession.fromBytes(base64Decode(encoded));
        final declining = _DecliningPairBackend();
        final expected = await graph.runAsync(
          inputs,
          backend: _IndividualProjectionBackend(),
        );
        final actual = await graph.runAsync(inputs, backend: declining);
        expect(declining.calls, exposed ? 0 : 1);
        for (final output in expected.keys) {
          expect(actual[output]!.toBytes(), expected[output]!.toBytes());
        }
      }
    },
  );

  test(
    'normalization fusion respects graph outputs and backend fallback',
    () async {
      final feeds = {
        'x': Tensor.float32([1, 2, 3], [1, 0, 2, -1, 0, -2]),
      };
      // Transpose, LayerNormalization, Transpose, Conv, GELU, Conv.
      for (final (encoded, exposed) in [
        (
          'CAk66AQKKwoBeBIKdHJhbnNwb3NlZCIJVHJhbnNwb3NlKg8KBHBlcm1AAEACQAGgAQcKRgoKdHJhbnNwb3NlZAoFc2NhbGUKBGJpYXMSAW4iEkxheWVyTm9ybWFsaXphdGlvbioUCgRheGlzGP///////////wGgAQIKKwoBbhIKbm9ybWFsaXplZCIJVHJhbnNwb3NlKg8KBHBlcm1AAEACQAGgAQcKIQoKbm9ybWFsaXplZAoCdzESAWUaBmV4cGFuZCIEQ29udgoTCgFlCgFkEgJkMBoCZDAiA0RpdgoTCgJkMBIDZXJmGgNlcmYiA0VyZgobCgNlcmYKA29uZRIEcGx1cxoEcGx1cyIDQWRkCiAKAWUKBHBsdXMSB3Byb2R1Y3QaB3Byb2R1Y3QiA011bAoiCgdwcm9kdWN0CgRoYWxmEgZoaWRkZW4aBGdlbHUiA011bAohCgZoaWRkZW4KAncyEgNvdXQaCGNvbnRyYWN0IgRDb252EgRwYWlyKi4IBAgCCAEQAUICdzFKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKi4IAggECAEQAUICdzJKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKg4QAUIEaGFsZkoEAAAAPyoNEAFCA29uZUoEAACAPyoLEAFCAWRKBPMEtT8qFQgCEAEiCAAAAEAAAAA/QgVzY2FsZSoUCAIQASIIAACAPgAAgD9CBGJpYXNaFwoBeBISChAIARIMCgIIAQoCCAIKAggDYhkKA291dBISChAIARIMCgIIAQoCCAIKAggDQgQKABAS',
          false,
        ),
        (
          'CAk6igUKKwoBeBIKdHJhbnNwb3NlZCIJVHJhbnNwb3NlKg8KBHBlcm1AAEACQAGgAQcKRgoKdHJhbnNwb3NlZAoFc2NhbGUKBGJpYXMSAW4iEkxheWVyTm9ybWFsaXphdGlvbioUCgRheGlzGP///////////wGgAQIKKwoBbhIKbm9ybWFsaXplZCIJVHJhbnNwb3NlKg8KBHBlcm1AAEACQAGgAQcKIQoKbm9ybWFsaXplZAoCdzESAWUaBmV4cGFuZCIEQ29udgoTCgFlCgFkEgJkMBoCZDAiA0RpdgoTCgJkMBIDZXJmGgNlcmYiA0VyZgobCgNlcmYKA29uZRIEcGx1cxoEcGx1cyIDQWRkCiAKAWUKBHBsdXMSB3Byb2R1Y3QaB3Byb2R1Y3QiA011bAoiCgdwcm9kdWN0CgRoYWxmEgZoaWRkZW4aBGdlbHUiA011bAohCgZoaWRkZW4KAncyEgNvdXQaCGNvbnRyYWN0IgRDb252EgRwYWlyKi4IBAgCCAEQAUICdzFKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKi4IAggECAEQAUICdzJKIAAAQMAAAADAAACAvwAAAAAAAIA/AAAAQAAAQEAAAIBAKg4QAUIEaGFsZkoEAAAAPyoNEAFCA29uZUoEAACAPyoLEAFCAWRKBPMEtT8qFQgCEAEiCAAAAEAAAAA/QgVzY2FsZSoUCAIQASIIAACAPgAAgD9CBGJpYXNaFwoBeBISChAIARIMCgIIAQoCCAIKAggDYhkKA291dBISChAIARIMCgIIAQoCCAIKAggDYiAKCm5vcm1hbGl6ZWQSEgoQCAESDAoCCAEKAggCCgIIA0IECgAQEg==',
          true,
        ),
      ]) {
        final graph = OnnxSession.fromBytes(base64Decode(encoded));
        final declining = _DecliningNormalizedBackend();
        final expected = await graph.runAsync(
          feeds,
          backend: _IndividualProjectionBackend(),
        );
        final result = await graph.runAsync(feeds, backend: declining);
        expect(declining.normalizedCalls, exposed ? 0 : 1);
        expect(declining.calls, 1);
        for (final key in expected.keys) {
          expect(result[key]!.toBytes(), expected[key]!.toBytes());
        }
      }
    },
  );

  test('paired native projections agree with two independent calls', () async {
    if (!backend.supportsPointwisePairs) return;
    const channels = 128, hidden = 1024, filters = 128, width = 13;
    final w1 = Tensor.float32([
      hidden,
      channels,
      1,
    ], List.generate(hidden * channels, (i) => (i % 13 - 6) / 64));
    final w2 = Tensor.float32([
      filters,
      hidden,
      1,
    ], List.generate(hidden * filters, (i) => (i % 17 - 8) / 128));
    final b1 = Tensor.float32([
      hidden,
    ], List.generate(hidden, (i) => (i % 13 - 5) / 32));
    final b2 = Tensor.float32([filters], List.generate(filters, (i) => i % 7));
    for (final repeated in [false, true]) {
      final x = Tensor.float32(
        [2, channels, width],
        List.generate(
          2 * channels * width,
          (i) => (((repeated ? i % (channels * width) : i) % 19) - 9) / 16,
        ),
      );
      final mid = await backend.pointwise(x, w1, b1, gelu: true);
      final expected = await backend.pointwise(
        Tensor.float32([2, hidden, width], mid!),
        w2,
        b2,
        gelu: false,
      );
      final actual = await backend.pointwisePair(x, w1, b1, w2, b2);
      expect(actual, orderedEquals(expected!));
    }
  });

  test('Metal computes a tail frame and preserves distinct batches', () async {
    if (!Platform.isMacOS && !Platform.isIOS) return;
    const channels = 256, hidden = 1024, width = 129;
    final metal = NativeMetalBackend();
    final x = Tensor.float32([
      2,
      channels,
      width,
    ], List.generate(2 * channels * width, (i) => (i % 23 - 11) / 16));
    final w1 = Tensor.float32([
      hidden,
      channels,
      1,
    ], List.generate(hidden * channels, (i) => (i % 29 - 14) / 512));
    final w2 = Tensor.float32([
      channels,
      hidden,
      1,
    ], List.generate(channels * hidden, (i) => (i % 13 - 6) / 256));
    final bias = Tensor.float32([
      hidden,
    ], List.generate(hidden, (i) => (i % 11 - 5) / 32));
    try {
      final expected = await backend.pointwisePair(x, w1, bias, w2, null);
      final result = await metal.pointwisePair(x, w1, bias, w2, null);
      expect(result, isNotNull);
      expect(metal.dispatches, greaterThan(0));
      expect(result!.length, expected!.length);
      var peak = 0.0;
      for (var i = 0; i < result.length; i++) {
        peak = math.max(peak, (expected[i] - result[i]).abs());
      }
      expect(peak, lessThan(1e-4));
    } finally {
      await metal.close();
    }
  });

  test('pointwise bias follows the rounded product before GELU', () async {
    const channels = 256, filters = 128, width = 9;
    final x = Tensor.float32(
      [1, channels, width],
      [
        ...List.filled(width, 16777216),
        ...List.filled(width, -16777216),
        ...List.filled((channels - 2) * width, 0),
      ],
    );
    final weights = Tensor.float32([
      filters,
      channels,
      1,
    ], List.filled(filters * channels, 1));
    final bias = Tensor.float32([filters], List.filled(filters, 1));
    final output = await backend.pointwise(x, weights, bias, gelu: false);
    expect(output, isNotNull);
    expect(output!, everyElement(1));
    final activated = await backend.pointwise(x, weights, bias, gelu: true);
    expect(activated, isNotNull);
    expect(activated!, everyElement(closeTo(0.84134475, 0.000001)));
  });
  test(
    'channel normalization preserves visible intermediates and frame tails',
    () async {
      final feeds = {
        'x': Tensor.float32([
          2,
          4,
          13,
        ], List.generate(104, (i) => (i % 17) / 7)),
        's': Tensor.float32([4], [1, 0.5, 2, -1]),
        'b': Tensor.float32([4], [-0.5, 0, 0.75, 4]),
      };
      for (final expose in [false, true]) {
        final graph = OnnxSession.fromBytes(
          base64Decode(
            expose
                ? 'CA46mAIKJAoBeBIDbmxjIglUcmFuc3Bvc2UqDwoEcGVybUAAQAJAAaABBwpNCgNubGMKAXMKAWISA3lsbiISTGF5ZXJOb3JtYWxpemF0aW9uKhQKBGF4aXMY////////////AaABAioRCgdlcHNpbG9uFazFJzegAQEKJAoDeWxuEgF5IglUcmFuc3Bvc2UqDwoEcGVybUAAQAJAAaABBxIMY2hhbm5lbCBub3JtWhcKAXgSEgoQCAESDAoCCAIKAggECgIIDVoPCgFzEgoKCAgBEgQKAggEWg8KAWISCgoICAESBAoCCARiFwoBeRISChAIARIMCgIIAgoCCAQKAggNYhkKA25sYxISChAIARIMCgIIAgoCCA0KAggEQgQKABAS'
                : 'CA46/QEKJAoBeBIDbmxjIglUcmFuc3Bvc2UqDwoEcGVybUAAQAJAAaABBwpNCgNubGMKAXMKAWISA3lsbiISTGF5ZXJOb3JtYWxpemF0aW9uKhQKBGF4aXMY////////////AaABAioRCgdlcHNpbG9uFazFJzegAQEKJAoDeWxuEgF5IglUcmFuc3Bvc2UqDwoEcGVybUAAQAJAAaABBxIMY2hhbm5lbCBub3JtWhcKAXgSEgoQCAESDAoCCAIKAggECgIIDVoPCgFzEgoKCAgBEgQKAggEWg8KAWISCgoICAESBAoCCARiFwoBeRISChAIARIMCgIIAgoCCAQKAggNQgQKABAS',
          ),
        );
        final expected = graph.run(feeds), nodes = <String>[];
        final outputs = await graph.runAsync(
          feeds,
          backend: backend,
          onAccelerated: nodes.add,
        );
        expect(nodes, hasLength(expose ? 0 : 1));
        for (final key in expected.keys) {
          expect(outputs[key]!.values, expected[key]!.values);
        }
      }
    },
  );

  test(
    'native depthwise valid convolution handles dilation and optional bias',
    () {
      final x = Tensor.float32([2, 3, 9], List.generate(54, (i) => i % 13 - 6));
      final w = Tensor.float32([3, 1, 3], List.generate(9, (i) => i - 4));
      for (final bias in [
        null,
        Tensor.float32([3], [0.5, 1, -2]),
      ]) {
        final y = backend.depthwiseValid(x, w, bias, 2);
        expect(y, [
          for (var row = 0; row < 6; row++)
            for (var t = 0; t < 5; t++)
              (bias?.values[row % 3] ?? 0) +
                  List.generate(
                    3,
                    (k) =>
                        x.values[row * 9 + t + 2 * k] *
                        w.values[(row % 3) * 3 + k],
                  ).reduce((a, b) => a + b),
        ]);
      }
    },
  );
  test(
    'scaling a residual rounds the product and preserves exposed edges',
    () async {
      final x = List<num>.generate(
        12,
        (i) => i % 6 < 2 ? 1 + 1 / 8388608 : i - 4,
      );
      final r = List<num>.generate(
        12,
        (i) => i % 6 < 2 ? -1 - 1 / 4194304 : 0.125 * i,
      );
      final feeds = {
        'x': Tensor.float32([2, 3, 2], x),
        'r': Tensor.float32([2, 3, 2], r),
      };
      for (final expose in [true, false]) {
        final graph = OnnxSession.fromBytes(
          base64Decode(
            expose
                ? 'CA46rQEKDgoBcwoBeBIBcCIDTXVsChAKAXIKAXASA291dCIDQWRkEghyZXNpZHVhbCoZCAEIAwgBEAEiDAEAgD8AAAA/AACAvkIBc1oXCgF4EhIKEAgBEgwKAggCCgIIAwoCCAJaFwoBchISChAIARIMCgIIAgoCCAMKAggCYhkKA291dBISChAIARIMCgIIAgoCCAMKAggCYhcKAXASEgoQCAESDAoCCAIKAggDCgIIAkIECgAQEg=='
                : 'CA46lAEKDgoBcwoBeBIBcCIDTXVsChAKAXIKAXASA291dCIDQWRkEghyZXNpZHVhbCoZCAEIAwgBEAEiDAEAgD8AAAA/AACAvkIBc1oXCgF4EhIKEAgBEgwKAggCCgIIAwoCCAJaFwoBchISChAIARIMCgIIAgoCCAMKAggCYhkKA291dBISChAIARIMCgIIAgoCCAMKAggCQgQKABAS',
          ),
        );
        final accelerated = <String>[];
        final y = await graph.runAsync(
          feeds,
          backend: backend,
          onAccelerated: accelerated.add,
        );
        final ref = graph.run(feeds);
        for (final key in ref.keys) expect(y[key]!.values, ref[key]!.values);
        expect(y['out']!.values.first, 0);
        expect(accelerated, hasLength(expose ? 0 : 1));
      }
    },
  );
  test('batched strided dilated convolution agrees with scalar sums', () async {
    final graph = OnnxSession.fromBytes(
      base64Decode(
        'CA46rQEKRgoBeAoBdwoEYmlhcxIBeSIEQ29udioQCglkaWxhdGlvbnNAAqABByoNCgRwYWRzQAFAAaABByoOCgdzdHJpZGVzQAKgAQcSBGNvbnZaFwoBeBISChAIARIMCgIIAgoCCAMKAggLWhcKAXcSEgoQCAESDAoCCAUKAggDCgIIA1oSCgRiaWFzEgoKCAgBEgQKAggFYhcKAXkSEgoQCAESDAoCCAIKAggFCgIIBUIECgAQEg==',
      ),
    );
    final x = List.generate(2 * 3 * 11, (i) => (i % 11 - 5).toDouble());
    final w = List.generate(5 * 3 * 3, (i) => (i % 5 - 2).toDouble());
    final feeds = {
      'x': Tensor.float32([2, 3, 11], x),
      'w': Tensor.float32([5, 3, 3], w),
      'bias': Tensor.float32([5], [0, 1, 2, 3, 4]),
    };
    final expected = <double>[];
    for (var b = 0; b < 2; b++) {
      for (var f = 0; f < 5; f++) {
        for (var t = 0; t < 5; t++) {
          var value = f.toDouble();
          for (var c = 0; c < 3; c++) {
            for (var p = 0; p < 3; p++) {
              final time = t * 2 + p * 2 - 1;
              if (time >= 0 && time < 11)
                value += w[(f * 3 + c) * 3 + p] * x[(b * 3 + c) * 11 + time];
            }
          }
          expected.add(value);
        }
      }
    }
    expect(graph.run(feeds)['y']!.values, expected);
    final accelerated = <String>[];
    final result = await graph.runAsync(
      feeds,
      backend: backend,
      onAccelerated: accelerated.add,
    );
    expect(result['y']!.values, expected);
    expect(accelerated, hasLength(1));
  });
}
