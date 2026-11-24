// Device-side client for tool/verify_models.py --browser. Model and reference
// data are served by the verification machine over a loopback HTTP connection.
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter_onnx/flutter_onnx.dart';

Future<void> main() => checkRemoteModels(Uri.base);
Future<void> checkRemoteModels(Uri server,
    {Future<Uint8List> Function(Uri)? readModel}) async {
  final client = http.Client();
  Uri url(String path) => server.resolve(path);
  OnnxSession? session;
  String? modelId;
  Map<String, Tensor> previous = {};
  try {
    while (true) {
      final request = jsonDecode((await client.get(url('/next'))).body);
      if (request == null) return;
      try {
        if (modelId != request['model']) {
          final model = readModel == null
              ? await client.readBytes(url('/model'))
              : await readModel(url('/model'));
          session = OnnxSession.fromBytes(model);
          modelId = request['model'];
          previous = {};
        }
        final feeds = <String, Tensor>{};
        for (final entry in (request['feeds'] as Map).entries) {
          final spec = entry.value;
          if (spec is String) {
            feeds[entry.key] = previous[spec]!;
          } else {
            final response = await client.get(url(spec['file']));
            final type =
                TensorType.values.firstWhere((t) => t.onnxId == spec['type']);
            feeds[entry.key] = Tensor.fromBytes(
                type, (spec['shape'] as List).cast<int>(), response.bodyBytes);
          }
        }
        previous = session!.run(feeds);
        final outputs = <String, Object>{};
        var index = 0;
        for (final entry in previous.entries) {
          final id = '/output/${index++}', t = entry.value;
          final uploaded = await client.post(url(id), body: t.toBytes());
          if (uploaded.statusCode != 200)
            throw StateError('Tensor upload failed');
          outputs[entry.key] = {
            'file': id,
            'type': t.type.onnxId,
            'shape': t.shape
          };
        }
        await client.post(url('/reply'),
            body: jsonEncode({'outputs': outputs}));
      } catch (error, stack) {
        await client.post(url('/reply'),
            body: jsonEncode(
                {'error': error.toString(), 'trace': stack.toString()}));
      }
    }
  } finally {
    client.close();
  }
}
