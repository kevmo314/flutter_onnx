// Run via "dart run test/supertonic_reference.dart <reference directory>".
// The fixtures contain RTChat's multilingual corpus synthesized by Supertone.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:flutter_onnx_cpu/flutter_onnx_cpu.dart';

Map readJson(String path) => jsonDecode(File(path).readAsStringSync()) as Map;

Iterable<num> flatten(Object value) sync* {
  if (value is num) {
    yield value;
  } else {
    for (final item in value as List) {
      yield* flatten(item);
    }
  }
}

Future<void> main(List<String> args) async {
  final root = args.single;
  final manifest = readJson('$root/manifest.json');
  final dir = manifest['model_dir'] as String;
  final graphs = {
    for (final name in [
      'duration_predictor',
      'text_encoder',
      'vector_estimator',
      'vocoder',
    ])
      name: OnnxSession.fromBytes(
        File('$dir/onnx/$name.onnx').readAsBytesSync(),
      ),
  };
  const cpu = NativeCpuBackend();
  final cases = (manifest['cases'] as List).cast<Map>();
  if (cases.isEmpty) throw StateError('Empty speech corpus');
  for (final example in cases) {
    Tensor fixture(String name) {
      final spec = example['tensors'][name];
      return Tensor.fromBytes(
        spec['dtype'] == 'int64' ? TensorType.int64 : TensorType.float32,
        (spec['shape'] as List).cast<int>(),
        File('$root/${spec['file']}').readAsBytesSync(),
      );
    }

    final styles = readJson('$dir/voice_styles/${example['voice']}.json');
    Tensor style(String name) => Tensor.float32(
      (styles[name]['dims'] as List).cast<int>(),
      flatten(styles[name]['data']),
    );
    Tensor mask(int width) =>
        Tensor.float32([1, 1, width], List.filled(width, 1));
    Tensor scalar(num value) => Tensor.float32([1], [value]);
    final ids = fixture('text_ids');
    final textMask = mask(ids.shape.last);
    final watch = Stopwatch()..start();
    final prediction = graphs['duration_predictor']!
        .run({
          'text_ids': ids,
          'text_mask': textMask,
          'style_dp': style('style_dp'),
        })['duration']!
        .values
        .single;
    final context = style('style_ttl');
    final text = await graphs['text_encoder']!.runAsync({
      'text_ids': ids,
      'text_mask': textMask,
      'style_ttl': context,
    }, backend: cpu);
    var latent = fixture('noise');
    final latentMask = mask(latent.shape.last);
    final steps = example['steps'] as int;
    for (var i = 0; i < steps; i++) {
      final out = await graphs['vector_estimator']!.runAsync({
        'noisy_latent': latent,
        'text_emb': text['text_emb']!,
        'style_ttl': context,
        'text_mask': textMask,
        'latent_mask': latentMask,
        'current_step': scalar(i),
        'total_step': scalar(steps),
      }, backend: cpu);
      latent = out['denoised_latent']!;
    }
    final wav = (await graphs['vocoder']!.runAsync({
      'latent': latent,
    }, backend: cpu))['wav_tts']!;
    watch.stop();
    final expected = Float32List.sublistView(
      File('$root/${example['pcm']}').readAsBytesSync(),
    );
    // Both the SDK and the mobile clients scale durations in float32.
    final seconds = Float32List.fromList([
      prediction / (example['speed'] as num),
    ]).single;
    final samples = Float32List.fromList([
      seconds * (example['sample_rate'] as int),
    ]).single.toInt();
    if (samples != expected.length || wav.length < samples) {
      throw StateError(
        "${example['name']}: expected ${expected.length} samples, got $samples (${wav.length} before trimming)",
      );
    }
    double peak = 0;
    for (var i = 0; i < samples; i++) {
      peak = math.max(peak, (wav.values[i] - expected[i]).abs().toDouble());
    }
    print(
      "${example['name']}: $samples samples, peak error $peak, ${watch.elapsed}",
    );
    if (!(peak < (manifest['tolerances']['wav'] as num))) {
      throw StateError("${example['name']}: PCM differs from Supertone");
    }
  }
}
