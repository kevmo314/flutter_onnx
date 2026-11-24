part of '../flutter_onnx.dart';

Tensor _generate(TensorType type, List<int> shape, num Function(int) f) {
  if (type == TensorType.float32) return _float(shape, (i) => f(i).toDouble());
  return Tensor.fromValues(type, shape, List.generate(_size(shape), f));
}

List<int> _indices(Tensor t) {
  _require(t.type == TensorType.int32 || t.type == TensorType.int64,
      'Expected integer indices');
  return t.values.cast<int>();
}

// ONNX permits INT64_MAX and INT64_MIN as open Slice boundaries. All legal
// tensor dimensions fit in 31 bits, so saturation here preserves the slice.
List<int> _sliceBounds(Tensor t) {
  if (t._wide == null) return _indices(t);
  final limit = BigInt.from(0x7fffffff) + BigInt.one;
  return t._wide
      .map((v) => v > limit
          ? limit.toInt()
          : v < -limit
              ? -limit.toInt()
              : v.toInt())
      .toList();
}

Tensor? _arrayOp(_Node n, List<Tensor?> args) {
  final x = args.first!, shape = x.shape;
  switch (n.op) {
    case 'Size':
      return Tensor.int64([], [x.length]);
    case 'ScatterND':
      final indices = args[1]!, updates = args[2]!, coords = _indices(indices);
      _require(indices.shape.isNotEmpty, 'ScatterND needs ranked indices');
      final depth = indices.shape.last,
          blocks = _size(indices.shape.take(indices.shape.length - 1));
      _require(
          depth > 0 && depth <= shape.length, 'Invalid ScatterND index depth');
      final chunk = _size(shape.skip(depth)), strides = _strides(shape);
      _require(
          updates.type == x.type && updates.values.length == blocks * chunk,
          'ScatterND updates mismatch');
      final copy = List<num>.of(x.values);
      for (var i = 0; i < blocks; i++) {
        var base = 0;
        for (var j = 0; j < depth; j++) {
          var v = coords[i * depth + j];
          if (v < 0) v += shape[j];
          _require(v >= 0 && v < shape[j], 'ScatterND index out of range');
          base += v * strides[j];
        }
        for (var j = 0; j < chunk; j++) {
          final old = copy[base + j], update = updates.values[i * chunk + j];
          copy[base + j] = switch (n.attrs['reduction'] ?? 'none') {
            'none' => update,
            'add' => old + update,
            'mul' => old * update,
            'min' => math.min(old, update),
            'max' => math.max(old, update),
            _ => throw UnsupportedError('ScatterND reduction')
          };
        }
      }
      return Tensor.fromValues(x.type, shape, copy);
    case 'Shape':
      final r = shape.length;
      int bound(int v) => (v < 0 ? r + v : v).clamp(0, r);
      final start = bound(n.integer('start', 0)),
          end = bound(n.integer('end', r));
      final dims = shape.sublist(start, math.max(start, end));
      return Tensor.int64([dims.length], dims);
    case 'Squeeze':
    case 'Unsqueeze':
      final expand = n.op == 'Unsqueeze';
      final axesTensor = args.elementAtOrNull(1);
      final axes = axesTensor == null
          ? [
              for (var i = 0; i < shape.length; i++)
                if (shape[i] == 1) i
            ]
          : _indices(axesTensor);
      final rank = expand ? shape.length + axes.length : shape.length;
      final set = {for (final a in axes) _axis(a, rank)};
      _require(set.length == axes.length, 'Duplicate axes');
      if (!expand) {
        _require(set.every((i) => shape[i] == 1),
            'Can only squeeze dimensions of length one');
        return x._reshape([
          for (var i = 0; i < rank; i++)
            if (!set.contains(i)) shape[i]
        ]);
      }
      var pos = 0;
      return x._reshape(
          [for (var i = 0; i < rank; i++) set.contains(i) ? 1 : shape[pos++]]);
    case 'Cast':
      return _cast(x, _type(n.integer('to', 0)), n);
    case 'ConstantOfShape':
      final value = n.attrs['value'] as Tensor?;
      if (value != null)
        _require(value.values.length == 1, 'ConstantOfShape expects one value');
      return _generate(value?.type ?? TensorType.float32, _indices(x),
          (i) => value?.values.single ?? 0);
    case 'Expand':
      final dims = _broadcastShape(shape, _indices(args[1]!));
      final indices = _IndexMap(shape, dims);
      return _reorder(x, dims, indices.at);
    case 'Tile':
      final repeats = _indices(args[1]!);
      _require(repeats.length == shape.length && repeats.every((v) => v >= 0),
          'Invalid repeats');
      final dst = [
        for (var i = 0; i < shape.length; i++) shape[i] * repeats[i]
      ];
      final strides = _strides(dst), src = _strides(shape);
      return _reorder(x, dst, (i) {
        var idx = 0;
        for (var a = 0; a < shape.length; a++) {
          idx += i ~/ strides[a] % dst[a] % shape[a] * src[a];
        }
        return idx;
      });
    case 'Slice':
      return _slice(x, _sliceBounds(args[1]!), _sliceBounds(args[2]!),
          args.elementAtOrNull(3), args.elementAtOrNull(4));
    case 'Gather':
    case 'GatherElements':
      final indices = args[1]!, ax = _axis(n.integer('axis', 0), shape.length);
      final values = _indices(indices);
      final inner = _size(shape.skip(ax + 1)), dim = shape[ax];
      int index(int i) {
        final v = values[i];
        _require(v >= -dim && v < dim, 'Out-of-bounds gather index $v');
        return v < 0 ? v + dim : v;
      }
      if (n.op == 'Gather') {
        return _reorder(
            x, [...shape.take(ax), ...indices.shape, ...shape.skip(ax + 1)],
            (i) {
          final p = i ~/ inner;
          return (p ~/ values.length * dim + index(p % values.length)) * inner +
              i % inner;
        });
      }
      final idxShape = indices.shape;
      _require(idxShape.length == shape.length, 'GatherElements rank mismatch');
      final istd = _strides(idxShape), xstd = _strides(shape);
      for (var j = 0; j < shape.length; j++) {
        _require(j == ax || idxShape[j] <= shape[j],
            'GatherElements shape mismatch');
      }
      return _reorder(x, idxShape, (i) {
        var p = 0;
        for (var j = 0; j < shape.length; j++) {
          p += (j == ax ? index(i) : i ~/ istd[j] % idxShape[j]) * xstd[j];
        }
        return p;
      });
    case 'Range':
      final start = x.values.single,
          limit = args[1]!.values.single,
          step = args[2]!.values.single;
      _require(step != 0, 'Zero step');
      return _generate(x.type, [math.max(0, ((limit - start) / step).ceil())],
          (i) => start + i * step);
    case 'Where':
      final a = args[1]!, b = args[2]!;
      _require(x.type == TensorType.boolean && a.type == b.type,
          'Invalid Where types');
      final dims = _broadcastShape(shape, _broadcastShape(a.shape, b.shape));
      final c = _IndexMap(shape, dims),
          ai = _IndexMap(a.shape, dims),
          bi = _IndexMap(b.shape, dims);
      return _generate(
          a.type,
          dims,
          (i) =>
              x.values[c.at(i)] != 0 ? a.values[ai.at(i)] : b.values[bi.at(i)]);
    case 'Clip':
      final min =
          args.elementAtOrNull(1)?.values.single ?? double.negativeInfinity;
      final max = args.elementAtOrNull(2)?.values.single ?? double.infinity;
      return _generate(
          x.type, shape, (i) => math.min(max, math.max(min, x.values[i])));
    case 'Trilu':
      _require(shape.length >= 2, 'Trilu expects matrices');
      final cols = shape.last,
          rows = shape[shape.length - 2],
          k = args.elementAtOrNull(1)?.values.single ?? 0;
      final upper = n.integer('upper', 1) != 0;
      return _generate(x.type, shape, (i) {
        final d = i % cols - i ~/ cols % rows;
        return (upper ? d >= k : d <= k) ? x.values[i] : 0;
      });
    case 'LogSoftmax':
      return _logSoftmax(x, n.integer('axis', -1));
    case 'Sum':
    case 'Mean':
    case 'Min':
    case 'Max':
      final tensors = args.cast<Tensor>();
      final dims =
          tensors.skip(1).fold(shape, (s, t) => _broadcastShape(s, t.shape));
      final maps = [for (final t in tensors) _IndexMap(t.shape, dims)];
      return _generate(x.type, dims, (i) {
        var acc = x.values[maps.first.at(i)];
        for (var j = 1; j < maps.length; j++) {
          final v = tensors[j].values[maps[j].at(i)];
          acc = switch (n.op) {
            'Min' => math.min(acc, v),
            'Max' => math.max(acc, v),
            _ => acc + v
          };
        }
        return n.op == 'Mean' ? acc / tensors.length : acc;
      });
  }
  if (n.op.startsWith('Reduce')) return _reduce(n, x, args.elementAtOrNull(1));
  final unary = _pointwise[n.op];
  if (unary != null)
    return _generate(
        n.op == 'IsNaN' || n.op == 'Not' ? TensorType.boolean : x.type,
        shape,
        (i) => unary(x.values[i].toDouble(), n));
  const comparisons = {
    'Equal',
    'Greater',
    'GreaterOrEqual',
    'Less',
    'LessOrEqual',
    'And',
    'Or',
    'Xor'
  };
  if (comparisons.contains(n.op) ||
      const {'Pow', 'PRelu'}.contains(n.op) ||
      (!x.type.floating && const {'Add', 'Mul', 'Div', 'Sub'}.contains(n.op))) {
    final b = args[1]!, dims = _broadcastShape(shape, b.shape);
    final ia = _IndexMap(shape, dims), ib = _IndexMap(b.shape, dims);
    if ((x.type == TensorType.int64 || x.type == TensorType.uint64) &&
        x.type == b.type &&
        !const {'Pow', 'PRelu'}.contains(n.op)) {
      final left = x.bigIntValues, right = b.bigIntValues;
      if (comparisons.contains(n.op)) {
        return _generate(TensorType.boolean, dims, (i) {
          final a = left[ia.at(i)], c = right[ib.at(i)];
          return (switch (n.op) {
            'Equal' => a == c,
            'Greater' => a > c,
            'Less' => a < c,
            'GreaterOrEqual' => a >= c,
            'LessOrEqual' => a <= c,
            _ => throw UnsupportedError(n.op)
          })
              ? 1
              : 0;
        });
      }
      return Tensor.bigInts(
          x.type,
          dims,
          List.generate(_size(dims), (i) {
            final a = left[ia.at(i)], c = right[ib.at(i)];
            final v = switch (n.op) {
              'Add' => a + c,
              'Sub' => a - c,
              'Mul' => a * c,
              'Div' => a ~/ c,
              _ => throw UnsupportedError(n.op)
            };
            return x.type == TensorType.int64
                ? v.toSigned(64)
                : v.toUnsigned(64);
          }));
    }
    return _generate(
        comparisons.contains(n.op) ? TensorType.boolean : x.type, dims, (i) {
      final a = x.values[ia.at(i)], c = b.values[ib.at(i)];
      return switch (n.op) {
        'Equal' => a == c ? 1 : 0,
        'Less' => a < c ? 1 : 0,
        'Greater' => a > c ? 1 : 0,
        'LessOrEqual' => a <= c ? 1 : 0,
        'GreaterOrEqual' => a >= c ? 1 : 0,
        'And' => a != 0 && c != 0 ? 1 : 0,
        'Or' => a != 0 || c != 0 ? 1 : 0,
        'Xor' => (a != 0) != (c != 0) ? 1 : 0,
        'Pow' => math.pow(a, c),
        'PRelu' => a < 0 ? a * c : a,
        'Mul' => a * c,
        'Add' => a + c,
        'Sub' => a - c,
        'Div' => a ~/ c,
        _ => throw StateError(n.op)
      };
    });
  }
  return null;
}

final _pointwise = <String, num Function(double, _Node)>{
  'HardSigmoid': (x, n) =>
      (n.real('alpha', 0.2) * x + n.real('beta', 0.5)).clamp(0, 1),
  'HardSwish': (x, n) => x * (x / 6 + 0.5).clamp(0, 1),
  'Swish': (x, n) => x / (1 + math.exp(-n.real('alpha', 1) * x)),
  'Reciprocal': (x, n) => 1 / x,
  'Floor': (x, n) => x.floorToDouble(),
  'Ceil': (x, n) => x.ceilToDouble(),
  'Round': (x, n) => !x.isFinite
      ? x
      : (x - x.floorToDouble() == 0.5
          ? (x.floor().isEven ? x.floor() : x.ceil())
          : x.round()),
  'Not': (x, n) => x == 0 ? 1 : 0,
  'IsNaN': (x, n) => x.isNaN ? 1 : 0,
  'Sign': (x, n) => x.sign,
  'Cos': (x, n) => math.cos(x),
  'Sin': (x, n) => math.sin(x),
  'Tan': (x, n) => math.tan(x),
  'Acos': (x, n) => math.acos(x),
  'Asin': (x, n) => math.asin(x),
  'Atan': (x, n) => math.atan(x),
  'Cosh': (x, n) => (math.exp(x) + math.exp(-x)) / 2,
  'Sinh': (x, n) => (math.exp(x) - math.exp(-x)) / 2,
  'Acosh': (x, n) => math.log(x + math.sqrt(x * x - 1)),
  'Asinh': (x, n) => x.sign * math.log(x.abs() + math.sqrt(1 + x * x)),
  'Atanh': (x, n) => 0.5 * math.log((1 + x) / (1 - x)),
  'Softsign': (x, n) => x / (1 + x.abs()),
  'Softplus': (x, n) => math.max(0, x) + math.log(1 + math.exp(-x.abs())),
  'Elu': (x, n) => x >= 0 ? x : n.real('alpha', 1) * (math.exp(x) - 1),
  'Celu': (x, n) =>
      x >= 0 ? x : n.real('alpha', 1) * (math.exp(x / n.real('alpha', 1)) - 1),
  'ThresholdedRelu': (x, n) => x > n.real('alpha', 1) ? x : 0,
  'Selu': (x, n) =>
      n.real('gamma', 1.0507009873554805) *
      (x >= 0 ? x : n.real('alpha', 1.6732632423543772) * (math.exp(x) - 1)),
};

Tensor _slice(Tensor x, List<int> starts, List<int> ends, Tensor? ax,
    Tensor? stepsTensor) {
  final axes =
      ax == null ? List.generate(starts.length, (i) => i) : _indices(ax);
  final steps = stepsTensor == null
      ? List.filled(axes.length, 1)
      : _sliceBounds(stepsTensor);
  _require([ends, axes, steps].every((a) => a.length == starts.length),
      'Slice input lengths differ');
  final dims = [...x.shape], strides = _strides(x.shape);
  var base = 0;
  final seen = <int>{};
  for (var i = 0; i < axes.length; i++) {
    final a = _axis(axes[i], dims.length), dim = x.shape[a], step = steps[i];
    _require(step != 0 && seen.add(a), 'Invalid Slice axes or steps');
    if (dim == 0) continue;
    int bound(int v, int lo, int hi) => (v < 0 ? v + dim : v).clamp(lo, hi);
    final start = bound(starts[i], 0, step > 0 ? dim : dim - 1),
        end = bound(ends[i], step > 0 ? 0 : -1, step > 0 ? dim : dim - 1);
    dims[a] = math.max(0, ((end - start) / step).ceil());
    base += start * strides[a];
    strides[a] *= step;
  }
  final dest = _strides(dims);
  return _reorder(x, dims, (i) {
    var p = base;
    for (var j = 0; j < dims.length; j++) {
      p += i ~/ dest[j] % dims[j] * strides[j];
    }
    return p;
  });
}

Tensor _reduce(_Node n, Tensor x, Tensor? axesTensor) {
  final raw = axesTensor == null ? n.ints('axes') : _indices(axesTensor);
  final axes = {for (final a in raw ?? <int>[]) _axis(a, x.shape.length)};
  if (axes.isEmpty) {
    if (n.integer('noop_with_empty_axes', 0) != 0) return x;
    axes.addAll(List.generate(x.shape.length, (i) => i));
  }
  final keep = n.integer('keepdims', 1) != 0;
  final dims = [
    for (var i = 0; i < x.shape.length; i++)
      if (!axes.contains(i)) x.shape[i] else if (keep) 1
  ];
  final strides = _strides(x.shape), outStrides = _strides(dims);
  final contribution = <int>[];
  var j = 0;
  for (var i = 0; i < x.shape.length; i++) {
    contribution.add(axes.contains(i) ? 0 : outStrides[j]);
    if (keep || !axes.contains(i)) j++;
  }
  final size = _size(dims), acc = Float64List(size);
  if (n.op == 'ReduceMax')
    acc.fillRange(
        0, size, x.type == TensorType.boolean ? 0 : double.negativeInfinity);
  if (n.op == 'ReduceMin')
    acc.fillRange(0, size, x.type == TensorType.boolean ? 1 : double.infinity);
  if (n.op == 'ReduceProd') acc.fillRange(0, size, 1);
  for (var i = 0; i < x.values.length; i++) {
    var dest = 0;
    for (var k = 0; k < contribution.length; k++) {
      if (contribution[k] != 0)
        dest += i ~/ strides[k] % x.shape[k] * contribution[k];
    }
    final v = x.values[i].toDouble();
    acc[dest] = switch (n.op) {
      'ReduceMax' => math.max(acc[dest], v),
      'ReduceMin' => math.min(acc[dest], v),
      'ReduceProd' => acc[dest] * v,
      'ReduceL1' => acc[dest] + v.abs(),
      'ReduceL2' || 'ReduceSumSquare' => acc[dest] + v * v,
      'ReduceLogSumExp' => acc[dest] + math.exp(v),
      _ => acc[dest] + v
    };
  }
  final count = _size([for (final a in axes) x.shape[a]]);
  return _generate(
      x.type,
      dims,
      (i) => switch (n.op) {
            'ReduceMean' => acc[i] / count,
            'ReduceL2' => math.sqrt(acc[i]),
            'ReduceLogSum' || 'ReduceLogSumExp' => math.log(acc[i]),
            _ => acc[i]
          });
}

Tensor _logSoftmax(Tensor x, int axis) {
  final a = _axis(axis, x.shape.length),
      inner = _size(x.shape.skip(a + 1)),
      width = x.shape[a];
  final out = Float32List(x.values.length);
  for (var base = 0; base < out.length; base += width * inner) {
    for (var j = 0; j < inner; j++) {
      var peak = double.negativeInfinity, sum = 0.0;
      for (var k = 0; k < width; k++) {
        peak = math.max(peak, x.values[base + j + k * inner].toDouble());
      }
      for (var k = 0; k < width; k++) {
        sum += math.exp(x.values[base + j + k * inner] - peak);
      }
      for (var k = 0; k < width; k++) {
        final i = base + j + k * inner;
        out[i] = x.values[i] - peak - math.log(sum);
      }
    }
  }
  return Tensor._(x.type, x.shape, out.asUnmodifiableView());
}
