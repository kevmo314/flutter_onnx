part of '../flutter_onnx.dart';

List<Tensor> _executeNode(_Node n, List<Tensor?> args) {
  final extra = _additionalOps(n, args);
  if (extra != null) return extra;
  final x = args.firstOrNull;
  switch (n.op) {
    case 'RandomUniform':
    case 'RandomUniformLike':
    case 'RandomNormal':
    case 'RandomNormalLike':
    case 'Multinomial':
    case 'Bernoulli':
      return [_random(n, args)];
    case 'TfIdfVectorizer':
      return [_vectorizer(n, x!)];
    case 'ImageDecoder':
      return [_decodeImage(n, x!)];
    case 'Col2Im':
    case 'ConvTranspose':
      return [_scatterWindows(n, args)];
    case 'AveragePool':
    case 'MaxPool':
    case 'LpPool':
      return _pool(n, x!);
    case 'MaxUnpool':
      return [_unpool(n, args)];
    case 'ConvInteger':
    case 'QLinearConv':
    case 'DeformConv':
      return [_convolve(n, args)];
    case 'Conv':
      if (x!.shape.length != 4 || x.type == TensorType.float64)
        return [_convolve(n, args)];
      return [_execute(n, args)];
    case 'BatchNormalization':
      if (n.integer('training_mode', 0) != 0)
        return _trainBatchNorm(n, args.cast<Tensor>());
      return [_execute(n, args)];
    case 'CausalConvWithState':
      return _causalConv(n, args);
    case 'LinearAttention':
      return _linearAttention(n, args);
    case 'RNN':
    case 'GRU':
    case 'LSTM':
      return _recurrent(n, args);
    case 'SoftmaxCrossEntropyLoss':
    case 'NegativeLogLikelihoodLoss':
      return _loss(n, args);
    case 'Mish':
      return [
        _generate(x!.type, x.shape, (i) {
          final v = x.values[i].toDouble();
          return v * _tanh(math.max(0, v) + math.log(1 + math.exp(-v.abs())));
        })
      ];
    case 'TensorScatter':
      return [_updateCache(n, args)];
    case 'NonMaxSuppression':
      return [_nms(n, args)];
    case 'Unique':
      return _unique(n, x!);
    case 'StringConcat':
    case 'StringNormalizer':
    case 'RegexFullMatch':
    case 'StringSplit':
      return _stringOp(n, args);
    case 'Equal':
      if (x!.type == TensorType.string) return _stringOp(n, args);
      return [_execute(n, args)];
    case 'com.microsoft::MatMulNBits':
      return [_matMulNBits(n, args)];
    case 'Attention':
      return _attention(n, args);
    case 'LayerNormalization':
    case 'RMSNormalization':
      return _normalize(n, args);
    case 'RotaryEmbedding':
      return [_rotary(n, args)];
    case 'Split':
      final ax = _axis(n.integer('axis', 0), x!.shape.length),
          dim = x.shape[ax],
          count = n.outputs.length;
      final split = args.elementAtOrNull(1);
      final step = (dim + count - 1) ~/ count;
      final sizes = split == null
          ? List.generate(
              count, (i) => math.min(step, math.max(0, dim - i * step)))
          : _indices(split);
      _require(
          sizes.length == count &&
              sizes.every((v) => v >= 0) &&
              sizes.fold(0, (a, b) => a + b) == dim,
          'Invalid split sizes');
      var start = 0;
      return [
        for (final size in sizes)
          _slice(x, [start], [start += size], Tensor.int64([1], [ax]), null)
      ];
    case 'ArgMax':
    case 'ArgMin':
    case 'TopK':
      return _select(n, args);
    case 'Dropout':
      if (args.elementAtOrNull(2)?.values.single == 1) {
        final ratio = args.elementAtOrNull(1)?.values.single ?? 0.5;
        _require(ratio >= 0 && ratio < 1, 'Invalid dropout ratio');
        final mask = _generate(TensorType.boolean, x!.shape,
            (_) => n.random.nextDouble() >= ratio ? 1 : 0);
        return [
          _generate(x.type, x.shape,
              (i) => mask.values[i] == 0 ? 0 : x.values[i] / (1 - ratio)),
          mask
        ];
      }
      return [
        x!,
        if (n.outputs.length > 1)
          _generate(TensorType.boolean, x.shape, (_) => 1)
      ];
    case 'SwiGLU':
      final b = args[1]!;
      _require(_sameShape(x!.shape, b.shape) && x.type == b.type,
          'SwiGLU shape/type mismatch');
      return [
        _generate(
            x.type,
            x.shape,
            (i) =>
                x.values[i] *
                b.values[i] /
                (1 + math.exp(-n.real('alpha', 1) * x.values[i])))
      ];
    case 'Gelu':
    case 'Erf':
      return [
        _generate(x!.type, x.shape, (i) {
          final v = x.values[i].toDouble();
          if (n.op == 'Erf') return _erf(v);
          if (n.attrs['approximate'] == 'tanh')
            return 0.5 *
                v *
                (1 +
                    _tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v * v * v)));
          return 0.5 * v * (1 + _erf(v / math.sqrt2));
        })
      ];
    default:
      return [_execute(n, args)];
  }
}

bool _sameShape(List<int> a, List<int> b) =>
    a.length == b.length && a.indexed.every((p) => p.$2 == b[p.$1]);
double _tanh(double x) {
  if (x.abs() < 1e-8) return x;
  final e = math.exp(-2 * x.abs());
  return x.sign * (1 - e) / (1 + e);
}

// Abramowitz & Stegun 7.1.26; maximum absolute approximation error 1.5e-7.
double _erf(double x) {
  if (x == 0) return x;
  final t = 1 / (1 + 0.3275911 * x.abs());
  final poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                  0.284496736) *
              t +
          0.254829592) *
      t;
  return x.sign * (1 - poly * math.exp(-x * x));
}

List<Tensor> _select(_Node n, List<Tensor?> args) {
  final x = args.first!, shape = [...x.shape];
  final top = n.op == 'TopK',
      largest = n.op == 'ArgMax' || (top && n.integer('largest', 1) != 0);
  final ax = _axis(n.integer('axis', top ? -1 : 0), shape.length),
      width = shape[ax],
      inner = _size(shape.skip(ax + 1));
  final k = top ? _indices(args[1]!).single : 1;
  _require(k >= 0 && k <= width, 'Invalid selection size');
  shape[ax] = k;
  final indices = List<int>.filled(_size(shape), 0);
  for (var row = 0; row < _size(shape.take(ax)); row++) {
    for (var col = 0; col < inner; col++) {
      num val(int i) => x.values[(row * width + i) * inner + col];
      final order = List.generate(width, (i) => i)
        ..sort((a, b) {
          final c = val(a).compareTo(val(b));
          if (c != 0) return largest ? -c : c;
          return n.integer('select_last_index', 0) != 0 ? b - a : a - b;
        });
      for (var j = 0; j < k; j++) {
        indices[(row * k + j) * inner + col] = order[j];
      }
    }
  }
  final result = Tensor.int64(shape, indices);
  if (top)
    return [
      _reorder(x, shape,
          (i) => (i ~/ (k * inner) * width + indices[i]) * inner + i % inner),
      result
    ];
  if (n.integer('keepdims', 1) == 0) shape.removeAt(ax);
  return [result._reshape(shape)];
}

List<Tensor> _normalize(_Node n, List<Tensor?> args) {
  final x = args[0]!, scale = args[1]!, bias = args.elementAtOrNull(2);
  final axis = _axis(n.integer('axis', -1), x.shape.length),
      width = _size(x.shape.skip(axis));
  final rms = n.op == 'RMSNormalization';
  final stash = _type(n.integer('stash_type', 1));
  final shape = [
    ...x.shape.take(axis),
    ...List.filled(x.shape.length - axis, 1)
  ];
  final groups = _size(shape),
      means = Float64List(groups),
      inv = Float64List(groups);
  for (var g = 0; g < groups; g++) {
    var mean = 0.0, sq = 0.0;
    if (!rms) {
      for (var i = 0; i < width; i++) {
        mean += x.values[g * width + i];
      }
      mean /= width;
    }
    for (var i = 0; i < width; i++) {
      final d = x.values[g * width + i] - mean;
      sq += d * d;
    }
    means[g] = mean;
    inv[g] = 1 / math.sqrt(sq / width + n.real('epsilon', 1e-5));
  }
  final sm = _IndexMap(scale.shape, x.shape),
      bm = bias == null ? null : _IndexMap(bias.shape, x.shape);
  final out = _generate(
      x.type,
      x.shape,
      (i) =>
          (x.values[i] - means[i ~/ width]) *
              inv[i ~/ width] *
              scale.values[sm.at(i)] +
          (bias?.values[bm!.at(i)] ?? 0));
  return [
    out,
    if (n.outputs.length > 1) Tensor.fromValues(stash, shape, means),
    if (n.outputs.length > 2) Tensor.fromValues(stash, shape, inv)
  ];
}

Tensor _rotary(_Node n, List<Tensor?> args) {
  final x = args[0]!,
      cos = args[1]!,
      sin = args[2]!,
      ids = args.elementAtOrNull(3);
  final rank = x.shape.length;
  _require(rank == 3 || rank == 4, 'RotaryEmbedding expects rank 3 or 4');
  final heads = rank == 4 ? x.shape[1] : n.integer('num_heads', 0);
  _require(heads > 0, 'Invalid head count');
  final seq = rank == 4 ? x.shape[2] : x.shape[1],
      width = rank == 4 ? x.shape[3] : x.shape[2] ~/ heads;
  final rotary = n.integer('rotary_embedding_dim', 0),
      dim = rotary == 0 ? width : rotary,
      half = dim ~/ 2;
  _require(dim <= width && dim % 2 == 0, 'Invalid rotary dimension');
  final interleaved = n.integer('interleaved', 0) != 0;
  return _generate(x.type, x.shape, (i) {
    final col = i % width;
    if (col >= dim) return x.values[i];
    final s = rank == 4 ? i ~/ width % seq : i ~/ (heads * width) % seq;
    final batch = i ~/ (seq * heads * width),
        p = ids == null ? batch * seq + s : ids.values[batch * seq + s].toInt();
    final index = p * cos.shape.last + (interleaved ? col ~/ 2 : col % half);
    final imag = interleaved ? col.isOdd : col >= half;
    final pair = i + (imag ? -1 : 1) * (interleaved ? 1 : half);
    return x.values[i] * cos.values[index] +
        (imag ? 1 : -1) * x.values[pair] * sin.values[index];
  });
}

List<Tensor> _attention(_Node n, List<Tensor?> args) {
  final packed = args[0]!.shape.length == 3;
  Tensor heads(Tensor t, int h) {
    if (!packed) {
      _require(t.shape.length == 4, 'Attention expects rank 3 or 4');
      return t;
    }
    _require(h > 0 && t.shape.last % h == 0, 'Invalid head count');
    return _transpose(
        t._reshape([t.shape[0], t.shape[1], h, t.shape.last ~/ h]),
        [0, 2, 1, 3]);
  }

  final q = heads(args[0]!, n.integer('q_num_heads', 0));
  var k = heads(args[1]!, n.integer('kv_num_heads', 0)),
      v = heads(args[2]!, n.integer('kv_num_heads', 0));
  final pastK = args.elementAtOrNull(4), pastV = args.elementAtOrNull(5);
  if (pastK != null) k = _concat([pastK, k], 2);
  if (pastV != null) v = _concat([pastV, v], 2);
  final [batch, hq, qs, depth] = q.shape;
  final hk = k.shape[1], ks = k.shape[2], dv = v.shape[3];
  _require(
      hk > 0 &&
          hq % hk == 0 &&
          k.shape[0] == batch &&
          k.shape.last == depth &&
          _sameShape(v.shape, [batch, hk, ks, dv]),
      'Incompatible attention shapes');
  final shape = [batch, hq, qs, ks],
      mask = args.elementAtOrNull(3),
      nonpad = args.elementAtOrNull(6);
  final maskWidth = mask == null || mask.shape.isEmpty ? ks : mask.shape.last;
  final maskShape = [batch, hq, qs, maskWidth],
      map = mask == null ? null : _IndexMap(mask.shape, maskShape);
  final left = n.integer('left_window_size', -1),
      right = n.integer('right_window_size', -1);
  final causal = n.integer('is_causal', 0) != 0,
      mode = n.integer('qk_matmul_output_mode', 0);
  final cap = n.real('softcap', 0),
      scale = n.real('scale', 1 / math.sqrt(depth));
  final scratch = ByteData(4);
  double rounded(double v, TensorType t) {
    if (t == TensorType.bfloat16 || t == TensorType.float16)
      return _decodeHalf(_encodeHalf(v, t, scratch), t, scratch);
    if (t == TensorType.float32) {
      scratch.setFloat32(0, v);
      return scratch.getFloat32(0);
    }
    return v;
  }

  final rootScale = rounded(math.sqrt(scale), q.type),
      precision = n.attrs.containsKey('softmax_precision')
          ? _type(n.integer('softmax_precision', 1))
          : q.type;
  final state = Float64List(ks), output = Float64List(batch * hq * qs * dv);
  final extra = n.outputs.length > 3 && n.outputs[3].isNotEmpty
      ? Float64List(_size(shape))
      : null;
  for (var b = 0; b < batch; b++) {
    final valid = nonpad?.values[b].toInt() ?? ks;
    final offset = pastK?.shape[2] ?? (nonpad == null ? 0 : valid - qs);
    for (var h = 0; h < hq; h++) {
      final kvBase = (b * hk + h ~/ (hq ~/ hk)) * ks;
      for (var s = 0; s < qs; s++) {
        final row = (b * hq + h) * qs + s;
        var peak = double.negativeInfinity;
        for (var j = 0; j < ks; j++) {
          var dot = 0.0;
          for (var d = 0; d < depth; d++)
            dot += rounded(q.values[row * depth + d] * rootScale, q.type) *
                rounded(k.values[(kvBase + j) * depth + d] * rootScale, q.type);
          final raw = rounded(dot, q.type),
              capped = cap > 0 ? rounded(cap * _tanh(raw / cap), q.type) : raw;
          final aligned = s + offset;
          final allowed = j < valid &&
              (!causal || j <= aligned) &&
              (left < 0 || j >= aligned - left) &&
              (right < 0 || j <= aligned + right);
          var bias = 0.0;
          if (mask != null && j < maskWidth) {
            final m = mask.values[map!.at(row * maskWidth + j)].toDouble();
            bias = mask.type == TensorType.boolean
                ? (m == 0 ? double.negativeInfinity : 0)
                : m;
          }
          final score = allowed && j < maskWidth
              ? capped + bias
              : double.negativeInfinity;
          state[j] = rounded(score, precision);
          peak = math.max(peak, state[j]);
          if (extra != null)
            extra[row * ks + j] = mode == 0
                ? raw
                : mode == 1
                    ? capped
                    : score;
        }
        var sum = 0.0;
        for (var j = 0; j < ks; j++) {
          state[j] = peak == double.negativeInfinity
              ? 0
              : rounded(
                  math.exp(rounded(state[j] - peak, precision)), precision);
          sum = precision == TensorType.bfloat16
              ? rounded(sum + state[j], precision)
              : sum + state[j];
        }
        for (var j = 0; j < ks; j++) {
          final prob = sum == 0
              ? 0.0
              : rounded(rounded(state[j] / rounded(sum, precision), precision),
                  q.type);
          if (extra != null && mode == 3) extra[row * ks + j] = prob;
          for (var d = 0; d < dv; d++) {
            output[row * dv + d] += prob * v.values[(kvBase + j) * dv + d];
          }
        }
      }
    }
  }
  var y = Tensor.fromValues(q.type, [batch, hq, qs, dv], output);
  if (packed) y = _transpose(y, [0, 2, 1, 3])._reshape([batch, qs, hq * dv]);
  return [y, k, v, if (extra != null) Tensor.fromValues(q.type, shape, extra)];
}

/// Four-bit blocks in this exporter are stored output-channel first, low nibble first.
Tensor _matMulNBits(_Node n, List<Tensor?> args) {
  final x = args[0]!,
      packed = args[1]!,
      scales = args[2]!,
      zero = args.elementAtOrNull(3);
  final k = n.integer('K', 0),
      cols = n.integer('N', 0),
      block = n.integer('block_size', 0);
  if (n.integer('bits', 0) != 4 ||
      block < 16 ||
      block.isOdd ||
      x.type != TensorType.float32 ||
      packed.type != TensorType.uint8 ||
      scales.type != TensorType.float32 ||
      (zero != null && zero.type != TensorType.uint8)) {
    throw UnsupportedError(
        'MatMulNBits expects FLOAT activations, four-bit blocks and optional packed zero points');
  }
  final groups = (k + block - 1) ~/ block, stride = block ~/ 2;
  _require(
      x.shape.isNotEmpty &&
          x.shape.last == k &&
          k > 0 &&
          cols > 0 &&
          packed.values.length == cols * groups * stride &&
          scales.values.length == cols * groups &&
          (zero == null || zero.values.length == cols * ((groups + 1) ~/ 2)),
      'Invalid quantized matrix dimensions');
  final a = x.values as Float32List,
      b = packed.values as Uint8List,
      f = scales.values as Float32List;
  return _float([...x.shape.take(x.shape.length - 1), cols], (i) {
    final col = i % cols, base = i ~/ cols * k;
    var sum = 0.0;
    for (var g = 0; g < groups; g++) {
      final idx = col * groups + g,
          offset = idx * stride,
          end = math.min(k, (g + 1) * block);
      final z = zero == null
          ? 8
          : ((zero.values[col * ((groups + 1) ~/ 2) + g ~/ 2].toInt() >>
                  (g % 2 * 4)) &
              15);
      var dot = 0.0;
      for (var d = g * block; d < end; d++) {
        final q = (b[offset + (d - g * block) ~/ 2] >> (d % 2 * 4)) & 15;
        dot += a[base + d] * (q - z);
      }
      sum += dot * f[idx];
    }
    return sum;
  });
}
