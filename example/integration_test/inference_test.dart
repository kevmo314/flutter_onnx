import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'checks.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  // XCTest can toggle accessibility while this headless inference check runs.
  setUpAll(() => binding.platformDispatcher.semanticsEnabledTestValue = false);
  tearDownAll(binding.platformDispatcher.clearSemanticsEnabledTestValue);
  testWidgets('cached attention, matrix multiplication and XOR',
      (tester) async {
    final result = await verifyInference();
    expect(result['xorCases'], 4);
  }, semanticsEnabled: false);
}
