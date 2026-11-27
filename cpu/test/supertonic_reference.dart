// Run via "dart run test/supertonic_reference.dart <reference directory>".
// The fixtures contain RTChat's multilingual corpus synthesized by Supertone.
import 'dart:convert';
import 'dart:io';
import 'dart:ffi';
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

// clock() counts CPU used by all process threads in POSIX microseconds.
final _processCpu = DynamicLibrary.process()
    .lookupFunction<Long Function(), int Function()>('clock');

Future<void> main(List<String> args) async {
  final playback = args.length == 2 && args.first == '--playback';
  final root = playback ? args.last : args.single;
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
  final cpu = Platform.environment['SPEECH_METAL'] == '1'
      ? NativeMetalBackend()
      : const NativeCpuBackend();
  final cases = (manifest['cases'] as List).cast<Map>();
  if (cases.isEmpty) throw StateError('Empty speech corpus');
  for (var round = 0; round < (playback ? 6 : 1); round++) {
    var cpuMicros = 0;
    var produced = Duration.zero;
    var finished = Duration.zero;
    var audio = Duration.zero;
    var first = Duration.zero;
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
      final durationStyle = style('style_dp'), context = style('style_ttl');
      var latent = fixture('noise');
      final cpuStart = playback ? _processCpu() : 0;
      final watch = Stopwatch()..start();
      final prediction = graphs['duration_predictor']!
          .run({
            'text_ids': ids,
            'text_mask': textMask,
            'style_dp': durationStyle,
          })['duration']!
          .values
          .single;
      final durationTime = watch.elapsed;
      final text = await graphs['text_encoder']!.runAsync({
        'text_ids': ids,
        'text_mask': textMask,
        'style_ttl': context,
      }, backend: cpu);
      final encoderTime = watch.elapsed;
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
      final denoiserTime = watch.elapsed;
      final wav = (await graphs['vocoder']!.runAsync({
        'latent': latent,
      }, backend: cpu))['wav_tts']!;
      watch.stop();
      if (playback) cpuMicros += _processCpu() - cpuStart;
      final expected = Float32List.sublistView(
        File(
          '$root/${playback ? example['tensors']['wav']['file'] : example['pcm']}',
        ).readAsBytesSync(),
      );
      // Both the SDK and the mobile clients scale durations in float32.
      final seconds = Float32List.fromList([
        prediction / (example['speed'] as num),
      ]).single;
      final samples = Float32List.fromList([
        seconds * (example['sample_rate'] as int),
      ]).single.toInt();
      final comparedSamples = playback ? wav.length : samples;
      if (comparedSamples != expected.length ||
          samples < 0 ||
          wav.length < samples) {
        throw StateError(
          "${example['name']}: expected ${expected.length} samples, got $samples (${wav.length} before trimming)",
        );
      }
      double peak = 0;
      for (var i = 0; i < comparedSamples; i++) {
        peak = math.max(peak, (wav.values[i] - expected[i]).abs().toDouble());
      }
      print(
        "${example['name']}: $samples samples, peak error $peak, ${watch.elapsed}",
      );
      if (!(peak < (manifest['tolerances']['wav'] as num))) {
        throw StateError("${example['name']}: PCM differs from Supertone");
      }
      if (playback) {
        final durationError = (seconds - fixture('duration').values.single)
            .abs();
        if (!(durationError < manifest['tolerances']['duration'])) {
          throw StateError(
            "${example['name']}: duration differs by $durationError seconds",
          );
        }
        final expectedSamples =
            File('$root/${example['pcm']}').lengthSync() ~/ 4;
        if (samples != expectedSamples) {
          throw StateError(
            "${example['name']}: trimmed $samples samples, expected $expectedSamples",
          );
        }
        final clip = Duration(
          microseconds: (samples * 1e6 / example['sample_rate']).round(),
        );
        produced += watch.elapsed;
        if (first == Duration.zero) first = produced;
        if (produced > finished) finished = produced;
        finished += clip;
        audio += clip;
        print(
          jsonEncode({
            'round': round + 1,
            'segment': example['name'],
            'duration_us': durationTime.inMicroseconds,
            'encoder_us': (encoderTime - durationTime).inMicroseconds,
            'denoiser_us': (denoiserTime - encoderTime).inMicroseconds,
            'vocoder_us': (watch.elapsed - denoiserTime).inMicroseconds,
            'trim_delta_samples': samples - expectedSamples,
          }),
        );
      }
    }
    if (playback) {
      if (cpu is NativeMetalBackend && cpu.dispatches == 0)
        throw StateError('No Metal projections executed');
      print(
        jsonEncode({
          'round': round + 1,
          'first_pcm_us': first.inMicroseconds,
          'synthesis_us': produced.inMicroseconds,
          'cpu_us': cpuMicros,
          if (cpu is NativeMetalBackend) 'metal_dispatches': cpu.dispatches,
          'audio_us': audio.inMicroseconds,
          'rtf': produced.inMicroseconds / audio.inMicroseconds,
          'estimated_playback_end_us': finished.inMicroseconds,
          'estimated_starvation_us': (finished - first - audio).inMicroseconds,
        }),
      );
    }
  }
}
