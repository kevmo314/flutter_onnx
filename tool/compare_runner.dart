// File-based verification driver for Python or other independent implementations.
import 'dart:convert';
import 'dart:io';
import 'package:flutter_onnx/flutter_onnx.dart';

void main(List<String> args) async {
  OnnxSession? session;
  String? current;
  Map<String, Tensor> last = {};
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
      final feeds = <String, Tensor>{};
      for (final entry in (request['feeds'] as Map).entries) {
        final spec = entry.value;
        if (spec is String) {
          feeds[entry.key] = last[spec]!;
          continue;
        }
        final shape = (spec['shape'] as List).cast<int>();
        final bytes = File(spec['file']).readAsBytesSync();
        final type = TensorType.values.firstWhere(
            (t) => (t == TensorType.boolean ? 'bool' : t.name) == spec['dtype'],
            orElse: () =>
                throw UnsupportedError('Input type ${spec['dtype']}'));
        feeds[entry.key] = Tensor.fromBytes(type, shape, bytes);
      }
      last = session!.run(feeds);
      final outputs = <String, Object>{};
      for (final (idx, e) in last.entries.indexed) {
        final path = '${request['directory']}/out-$idx.bin';
        final tensor = e.value;
        File(path).writeAsBytesSync(tensor.toBytes());
        outputs[e.key] = {
          'shape': tensor.shape,
          'dtype':
              tensor.type == TensorType.boolean ? 'bool' : tensor.type.name,
          'file': path
        };
      }
      stdout.writeln(jsonEncode({'outputs': outputs}));
    } catch (e, st) {
      stdout.writeln(jsonEncode({
        'error': e.toString(),
        'trace': '$st'.split('\n').take(4).join('\n')
      }));
    }
  }
}
