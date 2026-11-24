import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/widgets.dart';
import 'package:path_provider/path_provider.dart';
import 'package:on_device_tts/on_device_tts.dart';
import 'profile_playback.dart' as profile;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final dir = await getApplicationDocumentsDirectory();
  final result = File('${dir.path}/oracle.jsonl');
  final env = Platform.environment['ONNX_BNNS_SCOPE']!;
  void log(Map data) => result.writeAsStringSync('${jsonEncode(data)}\n',mode:FileMode.append,flush:true);
  try {
    final root = '${dir.path}/reference';
    final manifest = jsonDecode(File('$root/manifest.json').readAsStringSync()) as Map;
    final tts = await OnnxTts.load(modelDir:'${dir.path}/supertonic-3');
    var passed = true;
    try {
      for (final c in manifest['cases'] as List) {
        final audio = await tts.synthesize(c['text'] as String, language:c['lang'] as String,
          voiceId:c['voice'] as String, speed:(c['speed'] as num).toDouble(), steps:c['steps'] as int,
          noiseSeed:c['seed'] as int);
        final expected = Float32List.sublistView(File('$root/${c['pcm']}').readAsBytesSync());
        var error = 0.0;
        for (var i=0;i<math.min(audio.pcm.length,expected.length);i++) {
          error=math.max(error,(audio.pcm[i]-expected[i]).abs());
        }
        final correct = audio.pcm.length==expected.length && error<(manifest['tolerances']['wav'] as num);
        passed = passed && correct;
        log({'mode':env,'name':c['name'],'actual':audio.pcm.length,'expected':expected.length,'peak':error,'correct':correct});
      }
    } finally { await tts.unload(); }
    if (passed) await profile.main();
    log({'mode':env,'complete':true,'correct':passed});
  } catch (e,st) { log({'mode':env,'error':'$e','stack':'$st'}); }
}
