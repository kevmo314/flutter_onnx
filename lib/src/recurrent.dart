part of '../flutter_onnx.dart';

List<Tensor> _causalConv(_Node n, List<Tensor?> args) {
  final x = args[0]!,
      w = args[1]!,
      bias = args.elementAtOrNull(2),
      past = args.elementAtOrNull(3);
  final [b, c, t] = x.shape;
  final k = w.shape.last;
  _require(k > 0 && _sameShape(w.shape, [c, 1, k]),
      'Invalid CausalConvWithState weights');
  final prefix = past ?? _generate(x.type, [b, c, k - 1], (_) => 0);
  final joined = _concat([prefix, x], 2), stride = t + k - 1;
  final activation = n.attrs['activation'] ?? 'none';
  _require(const {'silu', 'swish', 'none'}.contains(activation),
      'Unknown causal convolution activation');
  final y = _generate(x.type, x.shape, (i) {
    final row = i ~/ t, idx = row * stride + i % t, ch = row % c;
    var sum = bias?.values[ch].toDouble() ?? 0.0;
    for (var j = 0; j < k; j++)
      sum += joined.values[idx + j] * w.values[ch * k + j];
    return activation == 'none' ? sum : sum / (1 + math.exp(-sum));
  });
  return [
    y,
    _slice(joined, [t], [stride], Tensor.int64([1], [2]), null)
  ];
}

List<Tensor> _linearAttention(_Node n, List<Tensor?> args) {
  final q = args[0]!,
      k = args[1]!,
      v = args[2]!,
      past = args.elementAtOrNull(3),
      g = args.elementAtOrNull(4),
      beta = args.elementAtOrNull(5);
  final hq = n.integer('q_num_heads', 0), hk = n.integer('kv_num_heads', 0);
  final [b, t, width] = q.shape;
  _require(hq > 0 && hk > 0 && hq % hk == 0 && width % hq == 0,
      'Invalid LinearAttention heads');
  final dk = width ~/ hq, dv = v.shape.last ~/ hk, group = hq ~/ hk;
  final rule = n.attrs['update_rule'] ?? 'gated_delta';
  _require(
      (g != null) == (rule == 'gated' || rule == 'gated_delta') &&
          (beta != null) == (rule == 'delta' || rule == 'gated_delta'),
      'Missing or unexpected gate parameters');
  final stateShape = [b, hk, dk, dv];
  final state = Float32List(_size(stateShape)),
      out = Float32List(b * t * hq * dv);
  if (past != null) {
    _require(_sameShape(past.shape, stateShape), 'Invalid recurrent state');
    state.setAll(0, past.values.cast<double>());
  }
  final attr = n.real('scale', 0), scale = attr == 0 ? 1 / math.sqrt(dk) : attr;
  for (var sample = 0; sample < b; sample++) {
    for (var time = 0; time < t; time++) {
      final row = sample * t + time;
      for (var h = 0; h < hk; h++) {
        final base = (sample * hk + h) * dk * dv,
            ki = (row * hk + h) * dk,
            vi = (row * hk + h) * dv;
        if (g != null) {
          final perHead = g.shape.last == hk;
          for (var a = 0; a < dk; a++) {
            final gate = math.exp(g.values[perHead ? row * hk + h : ki + a]);
            for (var j = 0; j < dv; j++) state[base + a * dv + j] *= gate;
          }
        }
        for (var j = 0; j < dv; j++) {
          var value = v.values[vi + j].toDouble();
          if (beta != null) {
            var predicted = 0.0;
            for (var a = 0; a < dk; a++)
              predicted += state[base + a * dv + j] * k.values[ki + a];
            value = (value - predicted) *
                beta.values[
                    row * beta.shape.last + (beta.shape.last == 1 ? 0 : h)];
          }
          for (var a = 0; a < dk; a++)
            state[base + a * dv + j] += k.values[ki + a] * value;
          for (var r = 0; r < group; r++) {
            final qi = (row * hq + h * group + r) * dk;
            var y = 0.0;
            for (var a = 0; a < dk; a++)
              y += q.values[qi + a] * state[base + a * dv + j];
            out[(row * hq + h * group + r) * dv + j] = scale * y;
          }
        }
      }
    }
  }
  return [
    Tensor.fromValues(q.type, [b, t, hq * dv], out),
    Tensor.fromValues(past?.type ?? q.type, stateShape, state)
  ];
}

Tensor _updateCache(_Node n, List<Tensor?> args) {
  final old = args[0]!,
      update = args[1]!,
      indices = args.elementAtOrNull(2),
      axis = _axis(n.integer('axis', -2), old.shape.length);
  final inner = _size(old.shape.skip(axis + 1)),
      maxLen = old.shape[axis],
      len = update.shape[axis];
  final batchStride = _size(update.shape.skip(1)),
      circular = n.attrs['mode'] == 'circular';
  final out = [...old.values];
  for (var i = 0; i < update.length; i++) {
    final b = i ~/ batchStride, start = indices?.values[b].toInt() ?? 0;
    var token = start + i ~/ inner % len;
    if (circular) token %= maxLen;
    _require(token >= 0 && token < maxLen, 'TensorScatter write exceeds cache');
    out[(i ~/ (len * inner) * maxLen + token) * inner + i % inner] =
        update.values[i];
  }
  return Tensor.fromValues(old.type, old.shape, out);
}

List<Tensor> _recurrent(_Node n, List<Tensor?> args) {
  final batchFirst = n.integer('layout', 0) == 1,
      gru = n.op == 'GRU',
      lstm = n.op == 'LSTM';
  final x = batchFirst ? _transpose(args[0]!, [1, 0, 2]) : args[0]!,
      w = args[1]!,
      r = args[2]!;
  final bias = args.elementAtOrNull(3),
      lengths = args.elementAtOrNull(4),
      p = args.elementAtOrNull(7);
  Tensor? initial(int i) {
    final v = args.elementAtOrNull(i);
    return v != null && batchFirst ? _transpose(v, [1, 0, 2]) : v;
  }

  final h0 = initial(5), c0 = initial(6);
  final [time, batch, features] = x.shape;
  final hidden = n.integer('hidden_size', r.shape.last),
      dirs = w.shape[0],
      gates = lstm
          ? 4
          : gru
              ? 3
              : 1;
  final acts = n.attrs['activations'] as List<String>?,
      alpha = n.attrs['activation_alpha'] as List<double>?,
      beta = n.attrs['activation_beta'] as List<double>?;
  final direction = n.attrs['direction'] ?? 'forward',
      clip = n.real('clip', double.infinity);
  final h = Float64List(dirs * batch * hidden),
      c = Float64List(h.length),
      y = Float64List(time * h.length);
  if (h0 != null) h.setAll(0, h0.values.cast<double>());
  if (c0 != null) c.setAll(0, c0.values.cast<double>());
  for (var dir = 0; dir < dirs; dir++) {
    double activate(int slot, double value, String fallback) {
      final i = dir *
              (lstm
                  ? 3
                  : gru
                      ? 2
                      : 1) +
          slot;
      final a = alpha?.elementAtOrNull(i),
          b = beta?.elementAtOrNull(i),
          v = value.clamp(-clip, clip);
      return switch (acts?.elementAtOrNull(i) ?? fallback) {
        'Tanh' => _tanh(v),
        'Sigmoid' => 1 / (1 + math.exp(-v)),
        'Relu' => math.max(0, v),
        'Affine' => (a ?? 1) * v + (b ?? 0),
        'LeakyRelu' => v < 0 ? v * (a ?? 0.01) : v,
        'ThresholdedRelu' => v > (a ?? 1) ? v : 0,
        'ScaledTanh' => (a ?? 1) * _tanh((b ?? 1) * v),
        'HardSigmoid' => ((a ?? 0.2) * v + (b ?? 0.5)).clamp(0, 1),
        'Elu' => v < 0 ? (a ?? 1) * (math.exp(v) - 1) : v,
        'Softsign' => v / (1 + v.abs()),
        'Softplus' => math.max(0, v) + math.log(1 + math.exp(-v.abs())),
        _ => throw UnsupportedError('Recurrent activation')
      };
    }

    for (var b = 0; b < batch; b++) {
      final size = lengths?.values[b].toInt() ?? time;
      _require(size >= 0 && size <= time, 'Invalid sequence length');
      final base = (dir * batch + b) * hidden;
      for (var step = 0; step < size; step++) {
        final t = (direction == 'reverse' ||
                (direction == 'bidirectional' && dir == 1))
            ? size - step - 1
            : step;
        final prev = h.sublist(base, base + hidden),
            input = Float64List(gates * hidden),
            rec = Float64List(gates * hidden);
        for (var g = 0; g < gates * hidden; g++) {
          final row = dir * gates * hidden + g, b0 = dir * gates * hidden * 2;
          var wi = bias?.values[b0 + g].toDouble() ?? 0.0,
              ri = bias?.values[b0 + gates * hidden + g].toDouble() ?? 0.0;
          for (var j = 0; j < features; j++)
            wi += w.values[row * features + j] *
                x.values[(t * batch + b) * features + j];
          for (var j = 0; j < hidden; j++)
            ri += r.values[row * hidden + j] * prev[j];
          input[g] = wi;
          rec[g] = ri;
        }
        double affine(int gate, int j) =>
            input[gate * hidden + j] + rec[gate * hidden + j];
        final reset = gru
            ? List.generate(hidden, (j) => activate(0, affine(1, j), 'Sigmoid'))
            : const <double>[];
        for (var j = 0; j < hidden; j++) {
          var next = 0.0;
          if (lstm) {
            final old = c[base + j], pi = dir * 3 * hidden + j;
            final i = activate(
                0, affine(0, j) + old * (p?.values[pi] ?? 0), 'Sigmoid');
            final f = n.integer('input_forget', 0) != 0
                ? 1 - i
                : activate(
                    0,
                    affine(2, j) + old * (p?.values[pi + 2 * hidden] ?? 0),
                    'Sigmoid');
            final cell = old * f + i * activate(1, affine(3, j), 'Tanh');
            c[base + j] = cell;
            next = activate(
                    0,
                    affine(1, j) + cell * (p?.values[pi + hidden] ?? 0),
                    'Sigmoid') *
                activate(2, cell, 'Tanh');
          } else if (gru) {
            final z = activate(0, affine(0, j), 'Sigmoid');
            var u = input[2 * hidden + j];
            if (n.integer('linear_before_reset', 0) != 0) {
              u += reset[j] * rec[2 * hidden + j];
            } else {
              u += bias?.values[dir * 6 * hidden + 5 * hidden + j] ?? 0;
              for (var k = 0; k < hidden; k++)
                u += r.values[((dir * 3 + 2) * hidden + j) * hidden + k] *
                    prev[k] *
                    reset[k];
            }
            next = (1 - z) * activate(1, u, 'Tanh') + z * prev[j];
          } else {
            next = activate(0, affine(0, j), 'Tanh');
          }
          h[base + j] = next;
          y[((t * dirs + dir) * batch + b) * hidden + j] = next;
        }
      }
    }
  }
  Tensor result(List<int> shape, Float64List vals, List<int> perm) {
    final t = Tensor.fromValues(x.type, shape, vals);
    return batchFirst ? _transpose(t, perm) : t;
  }

  return [
    result([time, dirs, batch, hidden], y, [2, 0, 1, 3]),
    result([dirs, batch, hidden], h, [1, 0, 2]),
    if (lstm) result([dirs, batch, hidden], c, [1, 0, 2])
  ];
}
