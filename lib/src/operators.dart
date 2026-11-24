part of '../flutter_onnx.dart';

Tensor _execute(_Node n, List<Tensor?> args) {
  if (n.op == 'Constant') {
    for (final e in n.attrs.entries) {
      final v = e.value;
      if (v is int) return Tensor.int64([], [v]);
      if (v is double) return Tensor.float32([], [v]);
      if (v is String) return Tensor.strings([], [v]);
      if (v is List<int>) return Tensor.int64([v.length], v);
      if (v is List<double>) return Tensor.float32([v.length], v);
      if (v is List<String>) return Tensor.strings([v.length], v);
    }
    final t = n.attrs['value'];
    if (t is! Tensor)
      throw UnsupportedError('${n.label}: expected tensor-valued Constant');
    return t;
  }
  final x = args[0]!;
  switch (n.op) {
    case 'Identity':
      return x;
    case 'Reshape':
      return _reshape(x, args[1]!, n.integer('allowzero', 0));
    case 'Transpose':
      return _transpose(x, n.ints('perm'));
    case 'Flatten':
      final axis = _axis(n.integer('axis', 1), x.shape.length, allowEnd: true);
      return x._reshape([_size(x.shape.take(axis)), _size(x.shape.skip(axis))]);
    case 'Concat':
      _require(n.attrs.containsKey('axis') && !args.contains(null),
          'Concat needs axis and nonempty input names');
      return _concat(args.cast<Tensor>(), n.integer('axis', 0));
  }
  final arrayResult = _arrayOp(n, args);
  if (arrayResult != null) return arrayResult;
  if (x.type == TensorType.float16 || x.type == TensorType.bfloat16) {
    final promoted = [
      for (final t in args)
        t == null
            ? null
            : t.type == x.type
                ? Tensor.float32(t.shape, t.values)
                : t
    ];
    final result = _execute(n, promoted);
    return Tensor.fromValues(x.type, result.shape, result.values);
  }
  for (final t in args.whereType<Tensor>()) {
    if (t.type != TensorType.float32)
      throw UnsupportedError('${n.label} requires FLOAT inputs');
  }
  switch (n.op) {
    case 'Add':
    case 'Sub':
    case 'Mul':
    case 'Div':
      return _binary(n.op, x, args[1]!);
    case 'MatMul':
      return _matmul(x, args[1]!);
    case 'Gemm':
      final b = args[1]!;
      _require(x.shape.length == 2 && b.shape.length == 2,
          'Gemm needs two matrices');
      final a1 = n.integer('transA', 0) == 0 ? x : _transpose(x, [1, 0]);
      final b1 = n.integer('transB', 0) == 0 ? b : _transpose(b, [1, 0]);
      return _matmul(a1, b1,
          alpha: n.real('alpha', 1),
          beta: n.real('beta', 1),
          bias: args.elementAtOrNull(2));
    case 'Softmax':
      return _softmax(x, n.integer('axis', -1));
    case 'Conv':
    case 'MaxPool':
    case 'AveragePool':
      return _spatial(n, args);
    case 'GlobalAveragePool':
      _require(x.shape.length >= 3, 'Expected N,C,spatial dimensions');
      final plane = _size(x.shape.skip(2));
      return _float([...x.shape.take(2), ...List.filled(x.shape.length - 2, 1)],
          (i) {
        var sum = 0.0;
        for (var j = 0; j < plane; j++) {
          sum += x.values[i * plane + j];
        }
        return sum / plane;
      });
    case 'BatchNormalization':
      _require(
          x.shape.length >= 2, 'BatchNormalization expects a channel axis');
      if (n.integer('training_mode', 0) != 0)
        throw UnsupportedError('${n.label}: training mode');
      final channels = x.shape[1];
      for (final t in args.skip(1).whereType<Tensor>()) {
        _require(t.shape.length == 1 && t.shape[0] == channels,
            'Incorrect BatchNormalization parameter size');
      }
      final stride = _size(x.shape.skip(2));
      final epsilon = n.real('epsilon', 1e-5);
      return _float(x.shape, (i) {
        final c = i ~/ stride % channels;
        return (x.values[i] - args[3]!.values[c]) /
                math.sqrt(args[4]!.values[c] + epsilon) *
                args[1]!.values[c] +
            args[2]!.values[c];
      });
    default:
      final double Function(double) fn = switch (n.op) {
        'Relu' => (v) => math.max(0.0, v),
        'Abs' => (v) => v.abs(),
        'Neg' => (v) => -v,
        'LeakyRelu' => (v) => v < 0 ? v * n.real('alpha', 0.01) : v,
        'Sigmoid' => (v) => 1 / (1 + math.exp(-v)),
        'Tanh' => (v) {
            if (v.abs() < 1e-8) return v;
            final e = math.exp(-2 * v.abs());
            return v.sign * (1 - e) / (1 + e);
          },
        'Sqrt' => math.sqrt,
        'Exp' => math.exp,
        'Log' => math.log,
        _ => throw UnsupportedError(n.label),
      };
      return _float(x.shape, (i) => fn(x.values[i].toDouble()));
  }
}

List<int> _broadcastShape(List<int> a, List<int> b) {
  final rank = math.max(a.length, b.length);
  return List.generate(rank, (i) {
    final ai = i - rank + a.length;
    final bi = i - rank + b.length;
    final ad = ai < 0 ? 1 : a[ai];
    final bd = bi < 0 ? 1 : b[bi];
    _require(ad == 1 || bd == 1 || ad == bd, 'Cannot broadcast $a with $b');
    return ad == 1 ? bd : ad;
  });
}

// Map output indices to an input after right alignment and broadcasting.
class _IndexMap {
  final List<int> outStrides, outShape, inStrides;
  _IndexMap(List<int> shape, this.outShape)
      : outStrides = _strides(outShape),
        inStrides = List.filled(outShape.length, 0) {
    final lead = outShape.length - shape.length;
    _require(lead >= 0, 'Cannot broadcast $shape to $outShape');
    final strides = _strides(shape);
    for (var i = 0; i < shape.length; i++) {
      _require(shape[i] == 1 || shape[i] == outShape[lead + i],
          'Cannot broadcast $shape to $outShape');
      if (shape[i] != 1) inStrides[lead + i] = strides[i];
    }
  }
  int at(int flat) {
    var offset = 0;
    for (var i = 0; i < inStrides.length; i++) {
      if (inStrides[i] != 0)
        offset += (flat ~/ outStrides[i] % outShape[i]) * inStrides[i];
    }
    return offset;
  }
}

Tensor _binary(String op, Tensor a, Tensor b) {
  final shape = _broadcastShape(a.shape, b.shape);
  final ia = _IndexMap(a.shape, shape), ib = _IndexMap(b.shape, shape);
  return _float(shape, (i) {
    final x = a.values[ia.at(i)].toDouble(), y = b.values[ib.at(i)].toDouble();
    return switch (op) {
      'Add' => x + y,
      'Sub' => x - y,
      'Mul' => x * y,
      'Div' => x / y,
      _ => throw StateError(op)
    };
  });
}

Tensor _matmul(Tensor a, Tensor b,
    {double alpha = 1, double beta = 1, Tensor? bias}) {
  _require(a.shape.isNotEmpty && b.shape.isNotEmpty, 'MatMul needs rank >= 1');
  final av = a.shape.length == 1, bv = b.shape.length == 1;
  final sa = [if (av) 1, ...a.shape], sb = [...b.shape, if (bv) 1];
  final m = sa[sa.length - 2], k = sa.last, n = sb.last;
  _require(k == sb[sb.length - 2], 'Matrix inner dimensions differ');
  final ba = sa.sublist(0, sa.length - 2), bb = sb.sublist(0, sb.length - 2);
  final batches = _broadcastShape(ba, bb);
  final ia = _IndexMap(ba, batches), ib = _IndexMap(bb, batches);
  final shape = [...batches, if (!av) m, if (!bv) n];
  final biasMap = bias == null ? null : _IndexMap(bias.shape, shape);
  final out = Float32List(_size(shape));
  final left = a.values as Float32List, right = b.values as Float32List;
  final row = Float64List(n);
  for (var batch = 0; batch < _size(batches); batch++) {
    final baseA = ia.at(batch) * m * k, baseB = ib.at(batch) * k * n;
    for (var r = 0; r < m; r++) {
      row.fillRange(0, n, 0);
      for (var d = 0; d < k; d++) {
        final scale = left[baseA + r * k + d], offset = baseB + d * n;
        for (var c = 0; c < n; c++) {
          row[c] += scale * right[offset + c];
        }
      }
      for (var c = 0; c < n; c++) {
        final i = (batch * m + r) * n + c;
        out[i] = alpha * row[c] +
            (bias == null || beta == 0
                ? 0
                : beta * bias.values[biasMap!.at(i)]);
      }
    }
  }
  return Tensor._(TensorType.float32, shape, out.asUnmodifiableView());
}

Tensor _softmax(Tensor x, int axis) {
  final ax = _axis(axis, x.shape.length);
  final width = x.shape[ax], inner = _size(x.shape.skip(ax + 1));
  final out = Float32List(x.values.length);
  if (out.isNotEmpty) {
    for (var base = 0; base < out.length; base += width * inner) {
      for (var j = 0; j < inner; j++) {
        var peak = double.negativeInfinity;
        for (var k = 0; k < width; k++) {
          peak = math.max(peak, x.values[base + j + k * inner].toDouble());
        }
        var sum = 0.0;
        for (var k = 0; k < width; k++) {
          sum += math.exp(x.values[base + j + k * inner] - peak);
        }
        for (var k = 0; k < width; k++) {
          final i = base + j + k * inner;
          out[i] = math.exp(x.values[i] - peak) / sum;
        }
      }
    }
  }
  return Tensor._(x.type, x.shape, out.asUnmodifiableView());
}

Tensor _reshape(Tensor x, Tensor pattern, int allowZero) {
  _require(pattern.type == TensorType.int64 && pattern.shape.length == 1,
      'Reshape needs a 1-D INT64 shape');
  final shape = pattern.values.cast<int>().toList();
  var infer = -1;
  for (var i = 0; i < shape.length; i++) {
    if (shape[i] == -1) {
      _require(infer == -1, 'Only one inferred dimension is allowed');
      infer = i;
      shape[i] = 1;
    } else if (shape[i] == 0 && allowZero == 0) {
      _require(i < x.shape.length, 'No input dimension to copy');
      shape[i] = x.shape[i];
    }
  }
  if (infer >= 0) {
    final known = _size(shape);
    _require(known > 0 && x.values.length % known == 0,
        'Ambiguous or incompatible inferred dimension');
    shape[infer] = x.values.length ~/ known;
  }
  return x._reshape(shape);
}

Tensor _transpose(Tensor x, List<int>? perm) {
  final rank = x.shape.length;
  perm ??= List.generate(rank, (i) => rank - 1 - i);
  _require(
      perm.length == rank &&
          perm.toSet().length == rank &&
          perm.every((i) => i >= 0 && i < rank),
      'Invalid permutation $perm');
  final shape = [for (final i in perm) x.shape[i]];
  final src = _strides(x.shape), dst = _strides(shape);
  final steps = [for (final i in perm) src[i]];
  return _reorder(x, shape, (flat) {
    var offset = 0;
    for (var i = 0; i < rank; i++) {
      offset += flat ~/ dst[i] % shape[i] * steps[i];
    }
    return offset;
  });
}

Tensor _concat(List<Tensor> tensors, int axis) {
  final first = tensors.first, shape = [...first.shape];
  final ax = _axis(axis, shape.length);
  shape[ax] = 0;
  for (final t in tensors) {
    _require(t.type == first.type && t.shape.length == shape.length,
        'Concat rank/type mismatch');
    for (var i = 0; i < shape.length; i++) {
      _require(i == ax || shape[i] == t.shape[i], 'Concat dimension mismatch');
    }
    shape[ax] += t.shape[ax];
  }
  final stride = _size(shape.skip(ax + 1));
  final outer = _size(shape.take(ax));
  Iterable<num> data() sync* {
    if (_size(shape) == 0) return;
    for (var row = 0; row < outer; row++) {
      for (final t in tensors) {
        final size = t.shape[ax] * stride;
        for (var i = row * size; i < (row + 1) * size; i++) {
          yield t.values[i];
        }
      }
    }
  }

  return Tensor.fromValues(first.type, shape, data());
}
