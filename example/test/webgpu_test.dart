@TestOn('browser')
library;

import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnx_accelerators/flutter_onnx_accelerators.dart';

void main() {
  test('WebGPU dispatch computes a rectangular product', () async {
    final gpu = await createMatMulBackend();
    if (gpu == null && !const bool.fromEnvironment('REQUIRE_WEBGPU')) {
      markTestSkipped('WebGPU unavailable in this browser');
      return;
    }
    expect(gpu, isNotNull);
    try {
      // The dispatch spans partial workgroups along both dimensions.
      final a = Float32List.fromList(
          List.generate(9 * 3, (i) => (i % 7 - 2).toDouble()));
      final b = Float32List.fromList(
          List.generate(3 * 11, (i) => (i % 5 - 1).toDouble()));
      final result = await gpu!.multiply(a, b, rows: 9, inner: 3, columns: 11);
      final expected = List.generate(9 * 11, (i) {
        var sum = 0.0;
        for (var k = 0; k < 3; k++) {
          sum += a[(i ~/ 11) * 3 + k] * b[k * 11 + i % 11];
        }
        return sum;
      });
      expect(result, expected);
    } finally {
      await gpu?.close();
    }
  });
}
