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
  Map<String, OnnxValue> previous = {};
  try {
    while (true) {
      final request = jsonDecode((await client.get(url('/next'))).body);
      if (request == null) return;
      try {
        if (modelId != request['model']) {
          final model = readModel == null
              ? await client.readBytes(url('/model'))
              : await readModel(url('/model'));
          final files = <String, Uint8List>{};
          for (final e in ((request['externalData'] as Map?) ?? {}).entries) {
            files[e.key] = await client.readBytes(url(e.value));
          }
          session = OnnxSession.fromBytes(model, externalData: files);
          modelId = request['model'];
          previous = {};
        }
        Future<OnnxValue> load(dynamic spec) async {
          if (spec == null) return const OnnxOptional(null);
          if (spec is String) return previous[spec]!;
          if (spec is List)
            return OnnxSequence(await Future.wait(spec.map(load)));
          final data = await client.readBytes(url(spec['file']));
          final type =
              TensorType.values.firstWhere((t) => t.onnxId == spec['type']);
          final shape = (spec['shape'] as List).cast<int>();
          return type == TensorType.string
              ? Tensor.strings(
                  shape, (jsonDecode(utf8.decode(data)) as List).cast<String>())
              : Tensor.fromBytes(type, shape, data);
        }

        final feeds = <String, OnnxValue>{};
        for (final e in (request['feeds'] as Map).entries)
          feeds[e.key] = await load(e.value);
        previous = session!.runValues(feeds);
        var index = 0;
        Future<Object?> save(OnnxValue value) async {
          if (value is OnnxOptional)
            return value.value == null ? null : await save(value.value!);
          if (value is OnnxSequence)
            return Future.wait(value.elements.map(save));
          final t = value as Tensor, id = '/output/${index++}';
          final data = t.type == TensorType.string
              ? utf8.encode(jsonEncode(t.stringValues))
              : t.toBytes();
          final uploaded = await client.post(url(id), body: data);
          if (uploaded.statusCode != 200)
            throw StateError('Tensor upload failed');
          return {'file': id, 'type': t.type.onnxId, 'shape': t.shape};
        }

        final outputs = <String, Object?>{};
        for (final entry in previous.entries)
          outputs[entry.key] = await save(entry.value);
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
