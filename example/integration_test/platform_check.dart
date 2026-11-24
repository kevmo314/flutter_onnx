import 'dart:convert';
import 'dart:io';
import 'package:flutter/widgets.dart';
import 'checks.dart';
import '../../tool/model_client.dart' show checkRemoteModels;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SizedBox.shrink());
  late final Map<String, Object?> result;
  try {
    result = {'ok': true, ...await verifyInference()};
  } catch (error, stack) {
    result = {
      'ok': false,
      'error': error.toString(),
      'stack': stack.toString()
    };
  }
  final file = File('${Directory.systemTemp.path}/flutter_onnx_check.json');
  final pending = await File('${file.path}.tmp')
      .writeAsString(jsonEncode(result), flush: true);
  await pending.rename(file.path);
  if (const bool.fromEnvironment('VERIFY_LARGE_MODELS') &&
      result['ok'] == true) {
    await checkRemoteModels(Uri.parse('http://127.0.0.1:17889'),
        readModel: (uri) async {
      final http = HttpClient();
      final temporary = File('${Directory.systemTemp.path}/full_model.onnx');
      try {
        final response = await (await http.getUrl(uri)).close();
        if (response.statusCode != 200)
          throw StateError('Model transfer failed');
        await response.pipe(temporary.openWrite());
        return await temporary.readAsBytes();
      } finally {
        http.close();
        if (await temporary.exists()) await temporary.delete();
      }
    });
  }
}
