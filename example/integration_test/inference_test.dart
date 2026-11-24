import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'checks.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  testWidgets('cached attention, matrix multiplication and XOR',
      (tester) async {
    final result = await verifyInference();
    expect(result['xorCases'], 4);
  });
}
