part of '../flutter_onnx.dart';

List<Tensor>? _additionalOps(_Node n, List<Tensor?> args) {
  final x = args.firstOrNull;
  if (x == null) return null;
  final shape = x.shape;
  switch (n.op) {
    case 'Resize':
    case 'Upsample':
      return [_resize(n, args)];
    case 'GridSample':
      return [_gridSample(n, x, args[1]!)];
    case 'AffineGrid':
      return [_affineGrid(n, x, args[1]!)];
    case 'RoiAlign':
    case 'MaxRoiPool':
      return [_roi(n, args)];
    case 'Einsum':
      return [_einsum(n, args.cast<Tensor>())];
    case 'DFT':
    case 'STFT':
    case 'MelWeightMatrix':
      return [_spectral(n, args)];
    case 'CastLike':
      return [_cast(x, args[1]!.type, n)];
    case 'BitCast':
      final type = _type(n.integer('to', 0));
      _require(type.bitWidth == x.type.bitWidth,
          'BitCast requires equal element widths');
      return [Tensor.fromBytes(type, shape, x.toBytes())];
    case 'BitwiseNot':
      return [_generate(x.type, shape, (i) => ~x.values[i].toInt())];
    case 'BitShift':
    case 'BitwiseAnd':
    case 'BitwiseOr':
    case 'BitwiseXor':
    case 'Mod':
      final b = args[1]!,
          dims = _broadcastShape(shape, b.shape),
          ia = _IndexMap(shape, dims),
          ib = _IndexMap(b.shape, dims);
      return [
        _generate(x.type, dims, (i) {
          final a = x.values[ia.at(i)], v = b.values[ib.at(i)];
          if (n.op == 'Mod') {
            if (n.integer('fmod', 0) != 0) return a.remainder(v);
            final r = a.remainder(v);
            return r != 0 && r.isNegative != v.isNegative ? r + v : r;
          }
          final lhs = BigInt.from(a.toInt()), rhs = BigInt.from(v.toInt());
          final bits = x.type.bitWidth;
          if (n.op == 'BitShift' &&
              (rhs.isNegative || rhs >= BigInt.from(bits)))
            return n.attrs['direction'] == 'RIGHT' && lhs.isNegative ? -1 : 0;
          final z = switch (n.op) {
            'BitwiseAnd' => lhs & rhs,
            'BitwiseOr' => lhs | rhs,
            'BitwiseXor' => lhs ^ rhs,
            _ => n.attrs['direction'] == 'LEFT'
                ? lhs << rhs.toInt()
                : lhs >> rhs.toInt()
          };
          return _portableInt(x.type.name.startsWith('int')
              ? z.toSigned(bits)
              : z.toUnsigned(bits));
        })
      ];
    case 'IsInf':
      return [
        _generate(TensorType.boolean, shape, (i) {
          final a = x.values[i];
          return ((a == double.infinity &&
                      n.integer('detect_positive', 1) != 0) ||
                  (a == double.negativeInfinity &&
                      n.integer('detect_negative', 1) != 0))
              ? 1
              : 0;
        })
      ];
    case 'Shrink':
      final b = n.real('bias', 0), lambda = n.real('lambd', 0.5);
      return [
        _generate(x.type, shape, (i) {
          final v = x.values[i];
          return v < -lambda
              ? v + b
              : v > lambda
                  ? v - b
                  : 0;
        })
      ];
    case 'EyeLike':
      _require(shape.length == 2, 'EyeLike requires a matrix');
      return [
        _generate(
            n.attrs.containsKey('dtype')
                ? _type(n.integer('dtype', 0))
                : x.type,
            shape,
            (i) => i % shape[1] - i ~/ shape[1] == n.integer('k', 0) ? 1 : 0)
      ];
    case 'CumSum':
    case 'CumProd':
      final axis = _axis(args[1]!.values.single.toInt(), shape.length),
          stride = _size(shape.skip(axis + 1)),
          width = shape[axis];
      final data = List<num>.filled(x.values.length, 0);
      final reverse = n.integer('reverse', 0) != 0,
          exclusive = n.integer('exclusive', 0) != 0,
          prod = n.op == 'CumProd';
      for (var g = 0; g < _size(shape.take(axis)); g++) {
        for (var c = 0; c < stride; c++) {
          num acc = prod ? 1 : 0;
          for (var k = 0; k < width; k++) {
            final t = reverse ? width - 1 - k : k,
                idx = (g * width + t) * stride + c;
            if (exclusive) data[idx] = acc;
            acc = prod ? acc * x.values[idx] : acc + x.values[idx];
            if (!exclusive) data[idx] = acc;
          }
        }
      }
      return [Tensor.fromValues(x.type, shape, data)];
    case 'NonZero':
      final coords = <int>[
            for (var i = 0; i < x.values.length; i++)
              if (x.values[i] != 0) i
          ],
          strides = _strides(shape);
      return [
        Tensor.int64([
          shape.length,
          coords.length
        ], [
          for (var a = 0; a < shape.length; a++)
            for (final i in coords) i ~/ strides[a] % shape[a]
        ])
      ];
    case 'OneHot':
      final depth = args[1]!.values.single.toInt(),
          v = args[2]!,
          axis = _axis(n.integer('axis', -1), shape.length + 1);
      final dims = [...shape]..insert(axis, depth);
      final stride = _size(shape.skip(axis));
      return [
        _generate(v.type, dims, (i) {
          var k = x.values[i ~/ (depth * stride) * stride + i % stride].toInt();
          if (k < 0) k += depth;
          return v.values[k == i ~/ stride % depth ? 1 : 0];
        })
      ];
    case 'GatherND':
      final ind = args[1]!,
          batch = n.integer('batch_dims', 0),
          depth = ind.shape.last,
          steps = _strides(shape);
      _require(
          batch >= 0 &&
              batch < ind.shape.length &&
              batch + depth <= shape.length &&
              _sameShape(
                  shape.take(batch).toList(), ind.shape.take(batch).toList()),
          'Invalid GatherND shapes');
      final chunk = _size(shape.skip(batch + depth)),
          indicesPerBatch =
              _size(ind.shape.skip(batch).take(ind.shape.length - 1 - batch)),
          batchSize = _size(shape.skip(batch));
      final dims = [
        ...ind.shape.take(ind.shape.length - 1),
        ...shape.skip(batch + depth)
      ];
      return [
        _reorder(x, dims, (i) {
          final block = i ~/ chunk;
          var offset = block ~/ indicesPerBatch * batchSize + i % chunk;
          for (var j = 0; j < depth; j++) {
            var p = ind.values[block * depth + j].toInt();
            final size = shape[j + batch];
            if (p < 0) p += size;
            _require(p >= 0 && p < size, 'GatherND index out of bounds');
            offset += p * steps[batch + j];
          }
          return offset;
        })
      ];
    case 'Compress':
      final condition = args[1]!, flatten = !n.attrs.containsKey('axis');
      final dims = flatten ? [x.values.length] : [...shape],
          axis = flatten ? 0 : _axis(n.integer('axis', 0), shape.length);
      final count = dims[axis],
          selected = [
            for (var i = 0; i < math.min(condition.values.length, count); i++)
              if (condition.values[i] != 0) i
          ];
      dims[axis] = selected.length;
      final stride = _size(dims.skip(axis + 1));
      return [
        _reorder(
            x,
            dims,
            (i) =>
                (i ~/ (stride * selected.length) * count +
                        selected[i ~/ stride % selected.length]) *
                    stride +
                i % stride)
      ];
    case 'ReverseSequence':
      final axis = _axis(n.integer('time_axis', 0), shape.length),
          batch = _axis(n.integer('batch_axis', 1), shape.length);
      final strides = _strides(shape), lengths = _indices(args[1]!);
      return [
        _reorder(x, shape, (i) {
          final p = i ~/ strides[axis] % shape[axis],
              length = lengths[i ~/ strides[batch] % shape[batch]];
          _require(
              length >= 0 && length <= shape[axis], 'Invalid sequence length');
          return p >= length ? i : i + (length - 1 - 2 * p) * strides[axis];
        })
      ];
    case 'DepthToSpace':
    case 'SpaceToDepth':
      final [b, c, h, w] = shape;
      final block = n.integer('blocksize', 0), bb = block * block;
      _require(block > 0, 'Invalid block size');
      if (n.op == 'SpaceToDepth') {
        _require(h % block == 0 && w % block == 0,
            'Input dimensions do not divide block size');
        final r = x._reshape([b, c, h ~/ block, block, w ~/ block, block]);
        return [
          _transpose(
                  r,
                  n.attrs['mode'] == 'CRD'
                      ? [0, 1, 3, 5, 2, 4]
                      : [0, 3, 5, 1, 2, 4])
              ._reshape([b, c * bb, h ~/ block, w ~/ block])
        ];
      }
      _require(c % bb == 0, 'Invalid depth');
      final dcr = (n.attrs['mode'] ?? 'DCR') == 'DCR';
      final r = x._reshape(dcr
          ? [b, block, block, c ~/ bb, h, w]
          : [b, c ~/ bb, block, block, h, w]);
      return [
        _transpose(r, dcr ? [0, 3, 4, 1, 5, 2] : [0, 1, 4, 2, 5, 3])
            ._reshape([b, c ~/ bb, h * block, w * block])
      ];
    case 'Scatter':
    case 'ScatterElements':
      final ind = args[1]!,
          updates = args[2]!,
          axis = _axis(n.integer('axis', 0), shape.length);
      final src = _strides(shape), idx = _strides(ind.shape);
      final copy = [...x.values];
      for (var i = 0; i < ind.values.length; i++) {
        var dest = 0;
        for (var d = 0; d < shape.length; d++) {
          var c =
              d == axis ? ind.values[i].toInt() : i ~/ idx[d] % ind.shape[d];
          if (c < 0) c += shape[d];
          _require(c >= 0 && c < shape[d], 'Scatter index out of range');
          dest += c * src[d];
        }
        final a = copy[dest], b = updates.values[i];
        copy[dest] = switch (n.attrs['reduction'] ?? 'none') {
          'none' => b,
          'add' => a + b,
          'mul' => a * b,
          'max' => math.max(a, b),
          'min' => math.min(a, b),
          _ => throw UnsupportedError('Scatter reduction')
        };
      }
      return [Tensor.fromValues(x.type, shape, copy)];
    case 'Pad':
    case 'CenterCropPad':
      return [_pad(n, args)];
    case 'HannWindow':
    case 'HammingWindow':
    case 'BlackmanWindow':
      final size = x.values.single.toInt(),
          denom = n.integer('periodic', 1) == 0 ? size - 1 : size;
      return [
        _generate(_type(n.integer('output_datatype', 1)), [size], (i) {
          final t = 2 * math.pi * i / denom;
          return switch (n.op) {
            'HannWindow' => 0.5 - 0.5 * math.cos(t),
            'HammingWindow' => 25 / 46 - 21 / 46 * math.cos(t),
            _ => 0.42 - 0.5 * math.cos(t) + 0.08 * math.cos(2 * t)
          };
        })
      ];
    case 'Hardmax':
      final ax = _axis(n.integer('axis', -1), shape.length),
          inner = _size(shape.skip(ax + 1)),
          w = shape[ax];
      final data = List<num>.filled(x.values.length, 0);
      for (var g = 0; g < _size(shape.take(ax)); g++) {
        for (var c = 0; c < inner; c++) {
          var best = 0;
          for (var j = 1; j < w; j++) {
            if (x.values[(g * w + j) * inner + c] >
                x.values[(g * w + best) * inner + c]) best = j;
          }
          if (w > 0) data[(g * w + best) * inner + c] = 1;
        }
      }
      return [Tensor.fromValues(x.type, shape, data)];
    case 'GlobalMaxPool':
    case 'GlobalLpPool':
      final size = _size(shape.skip(2)),
          p = n.integer('p', 2),
          maxPool = n.op == 'GlobalMaxPool';
      return [
        _generate(
            x.type, [...shape.take(2), ...List.filled(shape.length - 2, 1)],
            (i) {
          var value = maxPool ? double.negativeInfinity : 0.0;
          for (var j = 0; j < size; j++) {
            final v = x.values[i * size + j].toDouble();
            value = maxPool ? math.max(value, v) : value + math.pow(v.abs(), p);
          }
          return maxPool ? value : math.pow(value, 1 / p);
        })
      ];
    case 'LpNormalization':
      final ax = _axis(n.integer('axis', -1), shape.length),
          inner = _size(shape.skip(ax + 1)),
          w = shape[ax],
          p = n.integer('p', 2);
      return [
        _generate(x.type, shape, (i) {
          final start = i - (i ~/ inner % w) * inner;
          var sum = 0.0;
          for (var j = 0; j < w; j++)
            sum += math.pow(x.values[start + j * inner].abs(), p);
          return sum == 0 ? 0 : x.values[i] / math.pow(sum, 1 / p);
        })
      ];
    case 'GroupNormalization':
    case 'InstanceNormalization':
      final c = shape[1],
          groups =
              n.op == 'InstanceNormalization' ? c : n.integer('num_groups', 0);
      _require(groups > 0 && c % groups == 0, 'Invalid normalization groups');
      final spatial = _size(shape.skip(2)),
          size = (c ~/ groups) * spatial,
          scale = args[1]!,
          bias = args[2]!;
      final means = Float64List(shape[0] * groups),
          inv = Float64List(means.length);
      for (var i = 0; i < means.length; i++) {
        var m = 0.0, varSum = 0.0;
        for (var j = 0; j < size; j++) m += x.values[i * size + j];
        m /= size;
        for (var j = 0; j < size; j++) {
          final d = x.values[i * size + j] - m;
          varSum += d * d;
        }
        means[i] = m;
        inv[i] = 1 / math.sqrt(varSum / size + n.real('epsilon', 1e-5));
      }
      return [
        _generate(x.type, shape, (i) {
          final g = i ~/ size, ch = i ~/ spatial % c;
          return (x.values[i] - means[g]) * inv[g] * scale.values[ch] +
              bias.values[ch];
        })
      ];
    case 'MeanVarianceNormalization':
      final axes = n.ints('axes') ?? [0, 2, 3];
      final meanNode =
          _Node('ReduceMean', 'ReduceMean', ['mean'], ['x'], {'axes': axes});
      final m = _reduce(meanNode, x, null), map = _IndexMap(m.shape, shape);
      final square = _generate(x.type, shape, (i) {
        final d = x.values[i] - m.values[map.at(i)];
        return d * d;
      });
      final variance = _reduce(meanNode, square, null);
      return [
        _generate(
            x.type,
            shape,
            (i) =>
                (x.values[i] - m.values[map.at(i)]) /
                math.sqrt(variance.values[map.at(i)]))
      ];
    case 'LRN':
      final width = n.integer('size', 0),
          stride = _size(shape.skip(2)),
          channels = shape[1];
      return [
        _generate(x.type, shape, (i) {
          final c = i ~/ stride % channels;
          var sq = 0.0;
          for (var j = math.max(0, c - (width - 1) ~/ 2);
              j < math.min(channels, c + width ~/ 2 + 1);
              j++) {
            final v = x.values[i + (j - c) * stride];
            sq += v * v;
          }
          return x.values[i] /
              math.pow(n.real('bias', 1) + n.real('alpha', 0.0001) * sq / width,
                  n.real('beta', 0.75));
        })
      ];
    case 'Det':
      return [_det(x)];
    case 'MatMulInteger':
    case 'QLinearMatMul':
      return [_integerProduct(n, args)];
    case 'DynamicQuantizeLinear':
    case 'QuantizeLinear':
    case 'DequantizeLinear':
      return _quantize(n, args);
  }
  return null;
}

Tensor _pad(_Node n, List<Tensor?> args) {
  final x = args[0]!, rank = x.shape.length, center = n.op == 'CenterCropPad';
  final rawAxes = center
      ? n.ints('axes')
      : args.elementAtOrNull(3) == null
          ? null
          : _indices(args[3]!);
  final axes = rawAxes?.map((v) => _axis(v, rank)).toList() ??
      List.generate(rank, (i) => i);
  final widths = _indices(args[1]!),
      start = List<int>.filled(rank, 0),
      dims = [...x.shape];
  _require(widths.length == axes.length * (center ? 1 : 2),
      'Pad parameter lengths differ');
  for (var i = 0; i < axes.length; i++) {
    final a = axes[i];
    if (center) {
      start[a] = (widths[i] - dims[a]) ~/ 2;
      dims[a] = widths[i];
    } else {
      start[a] = widths[i];
      dims[a] += widths[i] + widths[i + axes.length];
    }
  }
  final mode = n.attrs['mode'] ?? 'constant',
      pad = args.elementAtOrNull(2)?.values.single ?? 0;
  if (x.type == TensorType.float32 &&
      (mode == 'constant' ||
          (mode == 'edge' && rank > 0 && x.shape.last > 0)) &&
      rank > 0 &&
      start.take(rank - 1).every((p) => p == 0) &&
      _sameShape(dims.sublist(0, rank - 1), x.shape.sublist(0, rank - 1))) {
    final output = Float32List(_size(dims));
    final value = pad.toDouble();
    if (value != 0 || value.isNegative)
      output.fillRange(0, output.length, value);
    final width = x.shape.last, dstWidth = dims.last;
    final input = x.values as Float32List;
    final first = (-start.last).clamp(0, width),
        dest = start.last.clamp(0, dstWidth);
    final copy = math.min(width - first, dstWidth - dest);
    for (var dst = 0, src = 0;
        dst < output.length;
        dst += dstWidth, src += width) {
      if (copy > 0)
        output.setRange(dst + dest, dst + dest + copy, input, src + first);
      if (mode == 'edge') {
        output.fillRange(dst, dst + dest, input[src]);
        output.fillRange(
            dst + dest + copy, dst + dstWidth, input[src + width - 1]);
      }
    }
    return Tensor._(x.type, dims, output.asUnmodifiableView());
  }
  final src = _strides(x.shape), dst = _strides(dims);
  return _generate(x.type, dims, (i) {
    var pos = 0;
    for (var d = 0; d < rank; d++) {
      var c = i ~/ dst[d] % dims[d] - start[d];
      final size = x.shape[d];
      if (c < 0 || c >= size) {
        if (mode == 'constant') return pad;
        if (mode == 'edge') {
          c = c.clamp(0, size - 1);
        } else if (mode == 'wrap') {
          c %= size;
        } else if (mode == 'reflect') {
          if (size == 1) {
            c = 0;
          } else {
            c %= 2 * (size - 1);
            c = math.min(c, 2 * (size - 1) - c);
          }
        } else {
          throw UnsupportedError('Pad mode $mode');
        }
      }
      pos += c * src[d];
    }
    return x.values[pos];
  });
}

Tensor _det(Tensor x) {
  final shape = x.shape, n = shape.last;
  _require(shape.length >= 2 && shape[shape.length - 2] == n,
      'Det expects square matrices');
  return _generate(x.type, shape.sublist(0, shape.length - 2), (batch) {
    final a = Float64List.fromList(x.values
        .skip(batch * n * n)
        .take(n * n)
        .map((v) => v.toDouble())
        .toList());
    var det = 1.0;
    for (var r = 0; r < n; r++) {
      var pivot = r;
      for (var j = r + 1; j < n; j++) {
        if (a[j * n + r].abs() > a[pivot * n + r].abs()) pivot = j;
      }
      if (a[pivot * n + r] == 0) return 0;
      if (pivot != r) {
        for (var j = r; j < n; j++) {
          final t = a[r * n + j];
          a[r * n + j] = a[pivot * n + j];
          a[pivot * n + j] = t;
        }
        det = -det;
      }
      det *= a[r * n + r];
      for (var j = r + 1; j < n; j++) {
        final f = a[j * n + r] / a[r * n + r];
        for (var c = r + 1; c < n; c++) a[j * n + c] -= f * a[r * n + c];
      }
    }
    return det;
  });
}

int _roundEven(double x) {
  final f = x.floor();
  return x - f == 0.5 ? (f.isEven ? f : f + 1) : x.round();
}

(int, int) _limits(TensorType t) => switch (t) {
      TensorType.uint4 => (0, 15),
      TensorType.int4 => (-8, 7),
      TensorType.uint2 => (0, 3),
      TensorType.int2 => (-2, 1),
      TensorType.uint8 => (0, 255),
      TensorType.int8 => (-128, 127),
      TensorType.uint16 => (0, 65535),
      TensorType.int16 => (-32768, 32767),
      TensorType.int32 => (-2147483648, 2147483647),
      _ => throw UnsupportedError('Quantizer output type $t')
    };

List<Tensor> _quantize(_Node n, List<Tensor?> args) {
  final x = args[0]!;
  if (n.op == 'DynamicQuantizeLinear') {
    final lo = math.min(0, x.values.fold<num>(0, math.min)).toDouble(),
        hi = math.max(0, x.values.fold<num>(0, math.max)).toDouble();
    final scale = Float32List.fromList([(hi == lo ? 1 : hi - lo) / 255]).single;
    final zp = _roundEven(-lo / scale).clamp(0, 255);
    return [
      _generate(
          TensorType.uint8,
          x.shape,
          (i) =>
              (_roundEven(Float32List.fromList([x.values[i] / scale]).single) +
                      zp)
                  .clamp(0, 255)),
      Tensor.float32([], [scale]),
      Tensor.fromValues(TensorType.uint8, [], [zp])
    ];
  }
  final s = args[1]!,
      z = args.elementAtOrNull(2),
      dq = n.op == 'DequantizeLinear';
  final output = n.integer('output_dtype', 0);
  final type = output != 0
      ? _type(output)
      : dq
          ? s.type
          : z?.type ?? TensorType.uint8;
  final axis =
      s.values.length == 1 ? 0 : _axis(n.integer('axis', 1), x.shape.length);
  final stride = _size(x.shape.skip(axis + 1)),
      block = n.integer('block_size', 0);
  final sstr = _strides(s.shape), xstr = _strides(x.shape);
  final limits = dq || type.floating ? (0, 0) : _limits(type);
  return [
    _generate(type, x.shape, (i) {
      var group = 0;
      if (block > 0) {
        for (var d = 0; d < x.shape.length; d++) {
          final c = i ~/ xstr[d] % x.shape[d];
          group += (d == axis ? c ~/ block : c) * sstr[d];
        }
      } else if (s.values.length > 1) {
        group = i ~/ stride % x.shape[axis];
      }
      final scale = s.values[group], zero = z?.values[group] ?? 0;
      if (!dq && type.floating) {
        final v = x.values[i] / scale;
        return type.compact
            ? _compactValue(type,
                _compactCode(type, v, saturate: n.integer('saturate', 1) != 0))
            : v;
      }
      return dq
          ? (x.values[i] - zero) * scale
          : (_roundEven(x.values[i] / scale) + zero)
              .clamp(limits.$1, limits.$2);
    })
  ];
}

Tensor _integerProduct(_Node n, List<Tensor?> args) {
  final quant = n.op == 'QLinearMatMul', a = args[0]!, b = args[quant ? 3 : 1]!;
  final az = args.elementAtOrNull(2), bz = args.elementAtOrNull(quant ? 5 : 3);
  final m = a.shape.length == 1 ? 1 : a.shape[a.shape.length - 2],
      cols = b.shape.length == 1 ? 1 : b.shape.last;
  final k = a.shape.last;
  Tensor centered(Tensor t, Tensor? z, bool left) {
    return _float(t.shape, (i) {
      final zero = z == null
          ? 0
          : z.values.length == 1
              ? z.values.single
              : z.values[left ? (i ~/ k) % m : i % cols];
      return (t.values[i] - zero).toDouble();
    });
  }

  final product = _matmul(centered(a, az, true), centered(b, bz, false));
  if (!quant)
    return Tensor.fromValues(TensorType.int32, product.shape, product.values);
  final yZero = args[7]!,
      limits = _limits(yZero.type),
      scale = args[1]!.values.single *
          args[4]!.values.single /
          args[6]!.values.single;
  return _generate(
      yZero.type,
      product.shape,
      (i) => (_roundEven(product.values[i] * scale) + yZero.values.single)
          .clamp(limits.$1, limits.$2));
}
