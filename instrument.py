from pathlib import Path
import sys
base=Path(sys.argv[1])
p=base/'lib/src/model.dart'
s=p.read_text().replace('    for (final node in plan) {','''    final totals = <String,int>{};
    for (final node in plan) {
      final lap = Stopwatch()..start();''').replace('    return frame.results().cast<String, Tensor>();','''    print('OPS ${outputs.keys.first} $totals');
    return frame.results().cast<String, Tensor>();''')
target='        frame.store(node, _valueNode(node, args, frame.values));\n      }'
s=s.replace(target,target+'''
      final x = args.firstOrNull;
      final key = node.op == 'Conv' && x is Tensor
        ? 'Conv(group=${node.integer('group',1)},kernel=${(args[1] as Tensor).shape.last})' : node.op;
      totals[key] = (totals[key] ?? 0) + lap.elapsedMicroseconds;''')
p.write_text(s)
p=base/'cpu/pubspec.yaml';p.write_text(p.read_text().replace('dev_dependencies:', 'dev_dependencies:\n  flutter_test:\n    sdk: flutter'))
p=base/'cpu/test/operator_profile_test.dart';p.write_text("""import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'supertonic_reference.dart' as corpus;
void main(){
  test('five-step audio profile', () async {
    await corpus.main(['--playback',Platform.environment['SPEECH_ROOT']!]);
  },timeout: const Timeout(Duration(minutes: 12)));
}
""")
