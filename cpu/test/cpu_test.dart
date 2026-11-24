import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:flutter_onnx_cpu/flutter_onnx_cpu.dart';
import 'package:test/test.dart';

void main() {
  const backend = NativeCpuBackend();
  test('matrix row and column tails include byte offsets', () async {
    for (final n in [1, 15, 16, 17, 33]) {
      const m = 5, k = 11;
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
  });
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
