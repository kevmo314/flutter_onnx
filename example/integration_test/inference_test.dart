import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_onnx/flutter_onnx.dart';
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
