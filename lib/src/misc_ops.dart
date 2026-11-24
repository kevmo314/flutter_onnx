part of '../flutter_onnx.dart';

Tensor _random(_Node n, List<Tensor?> args) {
  final x = args.firstOrNull, random = n.random;
  final type = _type(n.integer('dtype', x?.type.onnxId ?? 1));
  if (n.op == 'Multinomial') {
    final logits = x!,
        batch = x.shape[0],
        classes = x.shape[1],
        count = n.integer('sample_size', 1);
    final probabilities = Float64List(classes), out = <int>[];
    for (var b = 0; b < batch; b++) {
      var peak = double.negativeInfinity;
      for (var j = 0; j < classes; j++)
        peak = math.max(peak, logits.values[b * classes + j].toDouble());
      var sum = 0.0;
      for (var j = 0; j < classes; j++) {
        sum += math.exp(logits.values[b * classes + j] - peak);
        probabilities[j] = sum;
      }
      for (var i = 0; i < count; i++) {
        final p = random.nextDouble() * sum;
        out.add(probabilities.indexWhere((c) => c > p));
      }
    }
    return Tensor.fromValues(_type(n.integer('dtype', 6)), [batch, count], out);
  }
  if (n.op == 'Bernoulli')
    return _generate(
        type, x!.shape, (i) => random.nextDouble() < x.values[i] ? 1 : 0);
  final dims = n.ints('shape') ?? x!.shape;
  final lo = n.real('low', 0),
      hi = n.real('high', 1),
      mu = n.real('mean', 0),
      sd = n.real('scale', 1);
  return _generate(
      type,
      dims,
      (i) => n.op.startsWith('RandomUniform')
          ? lo + random.nextDouble() * (hi - lo)
          : mu +
              sd *
                  math.sqrt(-2 * math.log(1 - random.nextDouble())) *
                  math.cos(2 * math.pi * random.nextDouble()));
}

Tensor _vectorizer(_Node n, Tensor x) {
  final vocab = (n.attrs['pool_strings'] ?? n.attrs['pool_int64s']) as List;
  final counts = n.ints('ngram_counts')!,
      indices = n.ints('ngram_indexes')!,
      weights = n.attrs['weights'] as List<double>?;
  final width = indices.fold<int>(0, math.max) + 1,
      maxGram = n.integer('max_gram_length', 1),
      minGram = n.integer('min_gram_length', 1);
  final batch = x.shape.length < 2 ? 1 : x.shape[0],
      length = x.shape.isEmpty ? 1 : x.shape.last;
  final result = Float32List(batch * width),
      tokens = x.type == TensorType.string ? x.stringValues : x.values;
  var gramIndex = 0;
  for (var level = 0; level < counts.length; level++) {
    final len = level + 1,
        end = level + 1 < counts.length ? counts[level + 1] : vocab.length;
    for (var base = counts[level]; base < end; base += len, gramIndex++) {
      if (len < minGram || len > maxGram) continue;
      final out = indices[gramIndex];
      for (var b = 0; b < batch; b++) {
        var freq = 0;
        for (var step = 1;
            step <= (len == 1 ? 1 : n.integer('max_skip_count', 0) + 1);
            step++) {
          for (var pos = 0; pos + (len - 1) * step < length; pos++) {
            var match = true;
            for (var j = 0; j < len; j++)
              if (tokens[b * length + pos + j * step] != vocab[base + j]) {
                match = false;
                break;
              }
            if (match) freq++;
          }
        }
        result[b * width + out] += switch (n.attrs['mode']) {
          'IDF' => freq == 0 ? 0.0 : (weights?[out] ?? 1.0),
          'TFIDF' => freq * (weights?[out] ?? 1.0),
          _ => freq.toDouble()
        };
      }
    }
  }
  return Tensor.float32([if (x.shape.length == 2) batch, width], result);
}

Tensor _decodeImage(_Node n, Tensor x) {
  final bytes = Uint8List.fromList(x.values.cast<int>()),
      picture = img.decodeImage(bytes);
  if (picture == null)
    throw UnsupportedError('ImageDecoder: unsupported encoded image');
  final fmt = n.attrs['pixel_format'] ?? 'RGB',
      channels = fmt == 'Grayscale' ? 1 : 3;
  final out = Uint8List(picture.width * picture.height * channels);
  var i = 0;
  for (final pixel
      in picture.convert(format: img.Format.uint8, numChannels: 3)) {
    final r = pixel.r.toInt(), g = pixel.g.toInt(), b = pixel.b.toInt();
    if (channels == 1) {
      out[i++] = (r * 0.299 + g * 0.587 + b * 0.114).round();
    } else {
      out[i++] = fmt == 'BGR' ? b : r;
      out[i++] = g;
      out[i++] = fmt == 'BGR' ? r : b;
    }
  }
  return Tensor._(TensorType.uint8, [picture.height, picture.width, channels],
      out.asUnmodifiableView());
}

// MT19937 with the 53-bit uniform conversion used by NumPy RandomState.
// Keeping the state on the graph node advances it across session calls.
class _OnnxRandom {
  final _state = Uint32List(624);
  int _index = 624;
  _OnnxRandom(int? seed) {
    _state[0] = seed ?? math.Random().nextInt(0x100000000);
    final multiplier = BigInt.from(1812433253);
    for (var i = 1; i < _state.length; i++) {
      final x = _state[i - 1];
      _state[i] = (multiplier * BigInt.from(x ^ (x >>> 30)) + BigInt.from(i))
          .toUnsigned(32)
          .toInt();
    }
  }
  int _word() {
    if (_index == 624) {
      for (var i = 0; i < 624; i++) {
        final y =
            (_state[i] & 0x80000000) | (_state[(i + 1) % 624] & 0x7fffffff);
        _state[i] =
            _state[(i + 397) % 624] ^ (y >>> 1) ^ (y.isOdd ? 0x9908b0df : 0);
      }
      _index = 0;
    }
    var y = _state[_index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    return (y ^ (y >>> 18)).toUnsigned(32);
  }

  double nextDouble() =>
      ((_word() >>> 5) * 67108864.0 + (_word() >>> 6)) / 9007199254740992.0;
}
