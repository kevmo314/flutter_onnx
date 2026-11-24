part of '../flutter_onnx.dart';

// Spatial dimensions are ordered outermost to innermost, as in ONNX NCHW.
class _Window {
  final List<int> input, kernel, stride, dilation, pads, output;
  _Window(_Node n, this.input, this.kernel,
      {bool transpose = false, List<int>? specified})
      : stride = n.ints('strides') ?? List.filled(input.length, 1),
        dilation = n.ints('dilations') ?? List.filled(input.length, 1),
        pads = [...(n.ints('pads') ?? List.filled(2 * input.length, 0))],
        output = List.filled(input.length, 0) {
    final rank = input.length,
        auto = n.attrs['auto_pad'] ?? 'NOTSET',
        extra = n.ints('output_padding') ?? List.filled(rank, 0);
    _require(
        [kernel, stride, dilation, extra].every((s) => s.length == rank) &&
            pads.length == 2 * rank,
        'Window rank mismatch');
    for (var d = 0; d < rank; d++) {
      _require(kernel[d] > 0 && stride[d] > 0 && dilation[d] > 0,
          'Window sizes must be positive');
      final extent = 1 + (kernel[d] - 1) * dilation[d],
          same = auto == 'SAME_UPPER' || auto == 'SAME_LOWER';
      if (transpose) {
        final full = stride[d] * (input[d] - 1) + extra[d] + extent;
        output[d] = specified?[d] ??
            (same ? input[d] * stride[d] : full - pads[d] - pads[rank + d]);
        if (specified != null || same) {
          final p = full - output[d];
          pads[d] = (p + (auto == 'SAME_LOWER' ? 1 : 0)) ~/ 2;
          pads[d + rank] = p - pads[d];
        }
      } else if (same) {
        output[d] = (input[d] + stride[d] - 1) ~/ stride[d];
        final p = math.max(0, (output[d] - 1) * stride[d] + extent - input[d]);
        pads[d] = (p + (auto == 'SAME_LOWER' ? 1 : 0)) ~/ 2;
        pads[d + rank] = p - pads[d];
      } else {
        final size = (input[d] + pads[d] + pads[d + rank] - extent) / stride[d],
            ceil = n.integer('ceil_mode', 0) != 0;
        var out = (ceil ? size.ceil() : size.floor()) + 1;
        if (ceil && (out - 1) * stride[d] >= input[d] + pads[d]) out--;
        output[d] = math.max(0, out);
      }
    }
  }
}

List<int> _unravel(int flat, List<int> shape, List<int> strides) =>
    [for (var d = 0; d < shape.length; d++) flat ~/ strides[d] % shape[d]];
int _offset(List<int> coord, List<int> stride) {
  var p = 0;
  for (var d = 0; d < coord.length; d++) p += coord[d] * stride[d];
  return p;
}

List<Tensor> _pool(_Node n, Tensor x) {
  final dims = x.shape.sublist(2),
      kernel = n.ints('kernel_shape')!,
      win = _Window(n, dims, kernel),
      target = win.output;
  final src = _strides(dims),
      dst = _strides(target),
      ks = _strides(kernel),
      size = _size(dims),
      plane = _size(target);
  final cells = [
    for (var j = 0; j < _size(kernel); j++) _unravel(j, kernel, ks)
  ];
  final keep = n.outputs.length > 1 && n.outputs[1].isNotEmpty,
      indices = keep
          ? List<int>.filled(_size([...x.shape.take(2), ...target]), 0)
          : null;
  final isMax = n.op == 'MaxPool',
      lp = n.op == 'LpPool',
      p = n.integer('p', 2),
      include = n.integer('count_include_pad', 0) != 0;
  final shape = [...x.shape.take(2), ...target];
  final result = _generate(x.type, shape, (i) {
    final bc = i ~/ plane, out = _unravel(i % plane, target, dst);
    var val = isMax ? double.negativeInfinity : 0.0, count = 0;
    for (final cell in cells) {
      var off = 0, colMajor = 0, mul = 1, inside = true, padded = true;
      for (var d = 0; d < dims.length; d++) {
        final c =
            out[d] * win.stride[d] - win.pads[d] + cell[d] * win.dilation[d];
        inside = inside && c >= 0 && c < dims[d];
        padded = padded &&
            c >= -win.pads[d] &&
            c < dims[d] + win.pads[d + dims.length];
        off += c * src[d];
        colMajor += c * mul;
        mul *= dims[d];
      }
      if (!inside) {
        if (include && padded) count++;
        continue;
      }
      final xval = x.values[bc * size + off].toDouble();
      if (isMax) {
        if (xval > val || count == 0) {
          val = xval;
          if (indices != null)
            indices[i] = bc * size +
                (n.integer('storage_order', 0) == 0 ? off : colMajor);
        }
      } else {
        val += lp ? math.pow(xval.abs(), p) : xval;
      }
      count++;
    }
    return isMax
        ? val
        : lp
            ? math.pow(val, 1 / p)
            : val / count;
  });
  return [result, if (indices != null) Tensor.int64(shape, indices)];
}

Tensor _unpool(_Node n, List<Tensor?> args) {
  final x = args[0]!, idx = args[1]!, provided = args.elementAtOrNull(2);
  final win =
      _Window(n, x.shape.sublist(2), n.ints('kernel_shape')!, transpose: true);
  final shape = provided == null
      ? [...x.shape.take(2), ...win.output]
      : _indices(provided);
  final inferred = [...x.shape.take(2), ...win.output],
      src = _strides(inferred),
      dst = _strides(shape);
  final values = List<num>.filled(_size(shape), 0);
  for (var i = 0; i < x.length; i++) {
    final pos = _unravel(idx.values[i].toInt(), inferred, src);
    if (pos.indexed.every((p) => p.$2 < shape[p.$1]))
      values[_offset(pos, dst)] = x.values[i];
  }
  return Tensor.fromValues(x.type, shape, values);
}

Tensor _scatterWindows(_Node n, List<Tensor?> args) {
  final x = args[0]!,
      cols = n.op == 'Col2Im',
      w = cols ? null : args[1]!,
      bias = cols ? null : args.elementAtOrNull(2);
  final kernel = cols ? _indices(args[2]!) : w!.shape.sublist(2),
      target = cols ? _indices(args[1]!) : n.ints('output_shape');
  final rank = kernel.length, ks = _strides(kernel), kSize = _size(kernel);
  final source = cols ? _Window(n, target!, kernel).output : x.shape.sublist(2);
  final win = _Window(n, source, kernel,
          transpose: true, specified: cols ? null : target),
      out = target ?? win.output;
  final ss = _strides(source),
      ds = _strides(out),
      isize = _size(source),
      osize = _size(out),
      group = n.integer('group', 1);
  final cin = cols ? x.shape[1] ~/ kSize : x.shape[1],
      cout = cols ? cin : w!.shape[1] * group,
      ocg = cout ~/ group;
  final shape = [x.shape.first, cout, ...out],
      values = Float64List(_size(shape));
  if (bias != null)
    for (var i = 0; i < values.length; i++)
      values[i] = bias.values[i ~/ osize % cout].toDouble();
  for (var b = 0; b < shape.first; b++)
    for (var c = 0; c < cin; c++)
      for (var i = 0; i < isize; i++) {
        final pos = _unravel(i, source, ss);
        for (var k = 0; k < kSize; k++) {
          final tap = _unravel(k, kernel, ks);
          var flat = 0, valid = true;
          for (var d = 0; d < rank; d++) {
            final dst =
                pos[d] * win.stride[d] - win.pads[d] + tap[d] * win.dilation[d];
            if (dst < 0 || dst >= out[d]) {
              valid = false;
              break;
            }
            flat += dst * ds[d];
          }
          if (!valid) continue;
          if (cols) {
            values[(b * cout + c) * osize + flat] +=
                x.values[((b * cin + c) * kSize + k) * isize + i];
          } else {
            final v = x.values[(b * cin + c) * isize + i],
                start = c ~/ (cin ~/ group) * ocg;
            for (var j = 0; j < ocg; j++)
              values[(b * cout + start + j) * osize + flat] +=
                  v * w!.values[(c * ocg + j) * kSize + k];
          }
        }
      }
  return Tensor.fromValues(x.type, shape, values);
}

Tensor _convolve(_Node n, List<Tensor?> args) {
  final quant = n.op == 'QLinearConv',
      integer = n.op == 'ConvInteger',
      deform = n.op == 'DeformConv';
  final x = args[0]!,
      w = args[quant ? 3 : 1]!,
      bias = quant
          ? args.elementAtOrNull(8)
          : integer
              ? null
              : args.elementAtOrNull(deform ? 3 : 2);
  final offset = deform ? args[2] : null,
      mask = deform ? args.elementAtOrNull(4) : null;
  final xz = quant
          ? args[2]
          : integer
              ? args.elementAtOrNull(2)
              : null,
      wz = quant
          ? args[5]
          : integer
              ? args.elementAtOrNull(3)
              : null;
  final dims = x.shape.sublist(2),
      kernel = w.shape.sublist(2),
      win = _Window(n, dims, kernel),
      rank = dims.length;
  final input = _size(dims),
      plane = _size(win.output),
      ksize = _size(kernel),
      dst = _strides(win.output),
      src = _strides(dims),
      ks = _strides(kernel);
  final cin = x.shape[1],
      cout = w.shape[0],
      group = n.integer('group', 1),
      depth = w.shape[1],
      offsetGroups = n.integer('offset_group', 1);
  final shape = [x.shape[0], cout, ...win.output],
      type = quant
          ? args[7]!.type
          : integer
              ? TensorType.int32
              : x.type;
  return _generate(type, shape, (i) {
    final b = i ~/ (cout * plane),
        oc = i ~/ plane % cout,
        site = i % plane,
        pos = _unravel(site, win.output, dst),
        c0 = oc ~/ (cout ~/ group) * depth;
    var acc = bias?.values[oc].toDouble() ?? 0.0;
    for (var ic = 0; ic < depth; ic++)
      for (var tap = 0; tap < ksize; tap++) {
        final k = _unravel(tap, kernel, ks),
            coord = [
              for (var d = 0; d < rank; d++)
                pos[d] * win.stride[d] - win.pads[d] + k[d] * win.dilation[d]
            ];
        final base = (b * cin + c0 + ic) * input,
            og = (c0 + ic) ~/ (cin ~/ offsetGroups);
        double sample;
        if (offset != null) {
          final taps = [
            for (var d = 0; d < rank; d++)
              _taps(
                  coord[d] +
                      offset.values[((b * offsetGroups + og) * ksize * rank +
                                      tap * rank +
                                      d) *
                                  plane +
                              site]
                          .toDouble(),
                  'linear',
                  dims[d],
                  (c) => c < 0 || c >= dims[d] ? -1 : c)
          ];
          sample = _sample(x, base, src, taps) *
              (mask?.values[
                      ((b * offsetGroups + og) * ksize + tap) * plane + site] ??
                  1);
        } else {
          if (coord.indexed.any((e) => e.$2 < 0 || e.$2 >= dims[e.$1]))
            continue;
          sample =
              (x.values[base + _offset(coord, src)] - (xz?.values.single ?? 0))
                  .toDouble();
        }
        acc += sample *
            (w.values[(oc * depth + ic) * ksize + tap] -
                (wz?.values[wz.length == 1 ? 0 : oc] ?? 0));
      }
    if (!quant) return acc;
    final limits = _limits(type),
        scale = args[1]!.values.single *
            args[4]!.values[args[4]!.length == 1 ? 0 : oc] /
            args[6]!.values.single;
    return (_roundEven(acc * scale) + args[7]!.values.single)
        .clamp(limits.$1, limits.$2);
  });
}

List<Tensor> _trainBatchNorm(_Node n, List<Tensor> args) {
  final x = args[0],
      c = x.shape[1],
      inner = _size(x.shape.skip(2)),
      count = x.length ~/ c;
  final means = Float64List(c), vars = Float64List(c);
  for (var i = 0; i < x.length; i++)
    means[i ~/ inner % c] += x.values[i] / count;
  for (var i = 0; i < x.length; i++) {
    final ch = i ~/ inner % c, d = x.values[i] - means[ch];
    vars[ch] += d * d / count;
  }
  final eps = n.real('epsilon', 1e-5), m = n.real('momentum', 0.9);
  return [
    _generate(x.type, x.shape, (i) {
      final ch = i ~/ inner % c;
      return (x.values[i] - means[ch]) /
              math.sqrt(vars[ch] + eps) *
              args[1].values[ch] +
          args[2].values[ch];
    }),
    _generate(
        args[3].type, [c], (i) => m * args[3].values[i] + (1 - m) * means[i]),
    _generate(
        args[4].type, [c], (i) => m * args[4].values[i] + (1 - m) * vars[i])
  ];
}
