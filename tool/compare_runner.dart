// File-based verification driver for Python or other independent implementations.
import 'dart:convert';
import 'dart:io';
import 'package:flutter_onnx/flutter_onnx.dart';

void main(List<String> args) async {
  OnnxSession? session;
  String? current;
  Map<String, OnnxValue> last = {};
  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    try {
      final request = jsonDecode(line) as Map;
      final path = request['model'] as String;
      if (current != path) {
        session = null;
        last = {};
        current = null;
        session = OnnxSession.fromBytes(File(path).readAsBytesSync());
        current = path;
      }
      OnnxValue input(dynamic spec) {
        if (spec == null) return const OnnxOptional(null);
        if (spec is String) return last[spec]!;
        if (spec is List) return OnnxSequence(spec.map(input));
        final shape = (spec['shape'] as List).cast<int>(),
            file = File(spec['file']);
        final type = TensorType.values.firstWhere(
            (t) => t.onnxId == spec['type'],
            orElse: () => throw UnsupportedError('Input type ${spec['type']}'));
        return type == TensorType.string
            ? Tensor.strings(shape,
                (jsonDecode(file.readAsStringSync()) as List).cast<String>())
            : Tensor.fromBytes(type, shape, file.readAsBytesSync());
      }

      final feeds = <String, OnnxValue>{
        for (final e in (request['feeds'] as Map).entries) e.key: input(e.value)
      };
      last = session!.runValues(feeds);
      var count = 0;
      Object? output(OnnxValue v) {
        if (v is OnnxOptional) return v.value == null ? null : output(v.value!);
        if (v is OnnxSequence) return v.elements.map(output).toList();
        final tensor = v as Tensor,
            path = '${request['directory']}/out-${count++}.bin';
        if (tensor.type == TensorType.string) {
          File(path).writeAsStringSync(jsonEncode(tensor.stringValues));
        } else {
          File(path).writeAsBytesSync(tensor.toBytes());
        }
        return {
          'shape': tensor.shape,
          'type': tensor.type.onnxId,
          'file': path
        };
      }

      final outputs = {for (final e in last.entries) e.key: output(e.value)};
      stdout.writeln(jsonEncode({'outputs': outputs}));
    } catch (e, st) {
      stdout.writeln(jsonEncode({
        'error': e.toString(),
        'trace': '$st'.split('\n').take(4).join('\n')
      }));
    }
  }
}
