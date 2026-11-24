import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:flutter_onnx_cpu/flutter_onnx_cpu.dart';
import 'package:test/test.dart';

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
      const channels = 96, filters = 128, width = 13, batch = 2;
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
