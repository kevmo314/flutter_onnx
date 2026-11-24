import 'dart:convert';
import 'dart:io';
import 'package:flutter/widgets.dart';
import 'checks.dart';

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
}
