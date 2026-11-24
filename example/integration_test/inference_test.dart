import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnx_accelerators/flutter_onnx_accelerators.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('runs the bundled graph on the selected platform', (
    tester,
  ) async {
    final bytes = await rootBundle.load('assets/xor.onnx');
    final session = OnnxSession.fromBytes(
      bytes.buffer.asUint8List(bytes.offsetInBytes, bytes.lengthInBytes),
    );
    final backend = await createMatMulBackend();
    try {
      if (backend != null) {
        // Some NNAPI devices advertise acceleration but decline this operation.
        final values = await backend.multiply(
            Float32List.fromList([1, -2, 3, 0, 4, 2]),
            Float32List.fromList([2, 1, 0, 3, 4, -1]),
            rows: 2,
            inner: 3,
            columns: 2);
        if (values != null || backend.name == 'Metal Performance Shaders') {
          expect(values, [14, -8, 8, 10]);
          print('Verified matrix multiplication via ${backend.name}');
        }
      }
      for (final a in [0, 1]) {
        for (final b in [0, 1]) {
          final out = await session.runAsync({
            'bits': Tensor.float32([1, 2], [a, b]),
          }, backend: backend);
          expect(out['xor']!.values.single, closeTo((a ^ b).toDouble(), 1e-5));
        }
      }
    } finally {
      await backend?.close();
    }
  });
}
