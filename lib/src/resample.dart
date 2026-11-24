part of '../flutter_onnx.dart';

double _cubic(double x, double a) {
  x = x.abs();
  if (x <= 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  return 0;
}

// Each output point is the tensor product of the one-dimensional filter taps.
double _sample(
    Tensor x, int base, List<int> strides, List<List<(int, double)>> taps) {
  double visit(int axis, int offset) {
    if (axis == taps.length) return x.values[offset].toDouble();
    var sum = 0.0;
    for (final (p, weight) in taps[axis]) {
      if (p >= 0 && weight != 0)
        sum += weight * visit(axis + 1, offset + p * strides[axis]);
    }
    return sum;
  }

  return visit(0, base);
}

List<(int, double)> _taps(
    double p, String mode, int size, int Function(int) project,
    {double stretch = 1,
    double cubic = -0.75,
    bool exclude = false,
    String nearest = 'round_prefer_floor'}) {
  if (mode == 'nearest') {
    final pos = switch (nearest) {
      'floor' => p.floor(),
      'ceil' => p.ceil(),
      'round_prefer_ceil' => (p + 0.5).floor(),
      'even' => _roundEven(p),
      _ => (p - 0.5).ceil()
    };
    return [(project(pos), 1.0)];
  }
  final linear = mode == 'linear' || mode == 'bilinear';
  _require(linear || mode == 'cubic' || mode == 'bicubic',
      'Unknown interpolation mode $mode');
  final radius = (linear ? 1 : 2) * stretch;
  final taps = <(int, double)>[];
  var total = 0.0;
  for (var i = (p - radius).ceil(); i <= (p + radius).floor(); i++) {
    if (exclude && (i < 0 || i >= size)) continue;
    final d = (i - p) / stretch,
        w = linear ? math.max(0.0, 1 - d.abs()) : _cubic(d, cubic);
    if (w != 0) {
      total += w;
      taps.add((project(i), w));
    }
  }
  return [for (final (idx, w) in taps) (idx, w / total)];
}

Tensor _resize(_Node n, List<Tensor?> args) {
  final x = args[0]!, rank = x.shape.length, up = n.op == 'Upsample';
  final axes = (n.ints('axes') ?? List.generate(rank, (i) => i))
      .map((a) => _axis(a, rank))
      .toList();
  final roi = up ? null : args.elementAtOrNull(1);
  final scales = args.elementAtOrNull(up ? 1 : 2),
      sizes = up ? null : args.elementAtOrNull(3);
  final factors = List<double>.filled(rank, 1), dims = [...x.shape];
  _require((sizes?.length ?? 0) > 0 || (scales?.length ?? 0) > 0,
      'Resize requires scales or sizes');
  for (var i = 0; i < axes.length; i++) {
    final ax = axes[i];
    factors[ax] = (sizes?.length ?? 0) > 0
        ? sizes!.values[i] / x.shape[ax]
        : scales!.values[i].toDouble();
    dims[ax] = (sizes?.length ?? 0) > 0
        ? sizes!.values[i].toInt()
        : (x.shape[ax] * factors[ax]).floor();
  }
  final policy = n.attrs['keep_aspect_ratio_policy'] ?? 'stretch';
  if (policy != 'stretch') {
    final f = axes
        .map((a) => factors[a])
        .reduce(policy == 'not_larger' ? math.min : math.max);
    for (final a in axes) {
      factors[a] = f;
      dims[a] = (x.shape[a] * f).round();
    }
  }
  final mode = n.attrs['mode'] as String? ?? 'nearest',
      coordinate = n.attrs['coordinate_transformation_mode'] ??
          (up ? 'asymmetric' : 'half_pixel');
  final nearest = n.attrs['nearest_mode'] as String? ??
      (up ? 'floor' : 'round_prefer_floor');
  final all = <List<List<(int, double)>>>[], outside = <Set<int>>[];
  for (var d = 0; d < rank; d++) {
    final count = x.shape[d],
        out = dims[d],
        f = factors[d],
        scaled = count * f,
        index = axes.indexOf(d);
    final rs = index < 0 || roi == null || roi.length == 0
            ? 0
            : roi.values[index].toDouble(),
        re = index < 0 || roi == null || roi.length == 0
            ? 1
            : roi.values[index + axes.length].toDouble();
    final misses = <int>{};
    all.add(List.generate(out, (i) {
      final double p = index < 0
          ? i.toDouble()
          : switch (coordinate) {
              'align_corners' =>
                scaled <= 1 ? 0 : i * (count - 1) / (scaled - 1),
              'asymmetric' => i / f,
              'pytorch_half_pixel' => scaled > 1 ? (i + 0.5) / f - 0.5 : -0.5,
              'tf_half_pixel_for_nn' => (i + 0.5) / f,
              'tf_crop_and_resize' => scaled <= 1
                  ? (rs + re) * (count - 1) / 2
                  : rs * (count - 1) +
                      i * (re - rs) * (count - 1) / (scaled - 1),
              'half_pixel_symmetric' =>
                count / 2 * (1 - out / scaled) + (i + 0.5) / f - 0.5,
              'half_pixel' => (i + 0.5) / f - 0.5,
              _ => throw UnsupportedError('Resize coordinates $coordinate')
            };
      if (coordinate == 'tf_crop_and_resize' && (p < 0 || p > count - 1))
        misses.add(i);
      return _taps(p, mode, count, (c) => c.clamp(0, count - 1),
          stretch: n.integer('antialias', 0) != 0 ? math.max(1.0, 1 / f) : 1.0,
          cubic: n.real('cubic_coeff_a', -0.75),
          exclude: n.integer('exclude_outside', 0) != 0,
          nearest: nearest);
    }));
    outside.add(misses);
  }
  final dst = _strides(dims), src = _strides(x.shape);
  return _generate(x.type, dims, (i) {
    final positions = [for (var d = 0; d < rank; d++) i ~/ dst[d] % dims[d]];
    for (var d = 0; d < rank; d++) {
      if (outside[d].contains(positions[d]))
        return n.real('extrapolation_value', 0);
    }
    return _sample(
        x, 0, src, [for (var d = 0; d < rank; d++) all[d][positions[d]]]);
  });
}

Tensor _affineGrid(_Node n, Tensor theta, Tensor size) {
  final dims = _indices(size), rank = dims.length - 2;
  _require(rank == 2 || rank == 3, 'AffineGrid expects 2D or 3D images');
  final spatial = dims.sublist(2),
      strides = _strides(spatial),
      points = _size(spatial),
      aligned = n.integer('align_corners', 0) != 0;
  return _generate(theta.type, [dims[0], ...spatial, rank], (i) {
    final row = i % rank,
        point = i ~/ rank % points,
        b = i ~/ (rank * points),
        base = (b * rank + row) * (rank + 1);
    var y = theta.values[base + rank].toDouble();
    for (var a = 0; a < rank; a++) {
      final ax = rank - 1 - a,
          dim = spatial[ax],
          p = point ~/ strides[ax] % dim;
      final v = aligned
          ? (dim == 1 ? 0 : 2 * p / (dim - 1) - 1)
          : (2 * p + 1) / dim - 1;
      y += theta.values[base + a] * v;
    }
    return y;
  });
}

Tensor _gridSample(_Node n, Tensor x, Tensor grid) {
  final rank = x.shape.length - 2,
      c = x.shape[1],
      dims = grid.shape.sublist(1, grid.shape.length - 1);
  _require(
      rank == grid.shape.last &&
          dims.length == rank &&
          x.shape[0] == grid.shape[0],
      'Invalid GridSample shapes');
  final plane = _size(dims),
      input = x.shape.sublist(2),
      src = _strides(input),
      size = _size(input);
  final aligned = n.integer('align_corners', 0) != 0,
      pad = n.attrs['padding_mode'] ?? 'zeros',
      mode = n.attrs['mode'] as String? ?? 'linear';
  return _generate(x.type, [x.shape[0], c, ...dims], (i) {
    final b = i ~/ (c * plane), point = (b * plane + i % plane) * rank;
    final taps = <List<(int, double)>>[];
    for (var d = 0; d < rank; d++) {
      final dim = input[d],
          norm = grid.values[point + rank - 1 - d].toDouble(),
          p = aligned ? (norm + 1) * (dim - 1) / 2 : ((norm + 1) * dim - 1) / 2;
      int map(int j) {
        if (j >= 0 && j < dim) return j;
        if (pad == 'zeros') return -1;
        if (pad == 'border' || dim == 1) return j.clamp(0, dim - 1);
        final span = aligned ? dim - 1 : dim, lo = aligned ? 0.0 : -0.5;
        final dist = (j - lo) % (2 * span);
        return (lo + math.min(dist, 2 * span - dist)).round().clamp(0, dim - 1);
      }

      taps.add(_taps(p, mode, dim, map, nearest: 'even'));
    }
    return _sample(x, i ~/ plane * size, src, taps);
  });
}

Tensor _roi(_Node n, List<Tensor?> args) {
  final x = args[0]!,
      rois = args[1]!,
      pool = n.op == 'MaxRoiPool',
      h = x.shape[2],
      w = x.shape[3],
      channels = x.shape[1];
  final dims = n.ints('pooled_shape'),
      oh = dims?[0] ?? n.integer('output_height', 1),
      ow = dims?[1] ?? n.integer('output_width', 1);
  final scale = n.real('spatial_scale', 1),
      half = !pool &&
          (n.attrs['coordinate_transformation_mode'] ?? 'half_pixel') ==
              'half_pixel';
  return _generate(x.type, [rois.shape[0], channels, oh, ow], (i) {
    final r = i ~/ (channels * oh * ow),
        c = i ~/ (oh * ow) % channels,
        dy = i ~/ ow % oh,
        dx = i % ow;
    final start = r * (pool ? 5 : 4) + (pool ? 1 : 0),
        b = pool ? rois.values[start - 1].toInt() : args[2]!.values[r].toInt();
    double coord(int k) {
      final a = rois.values[start + k] * scale;
      return pool ? a.roundToDouble() : a - (half ? 0.5 : 0);
    }

    final x0 = coord(0),
        y0 = coord(1),
        rw = math.max(half ? 0.0 : 1.0, coord(2) - x0 + (pool ? 1 : 0)),
        rh = math.max(half ? 0.0 : 1.0, coord(3) - y0 + (pool ? 1 : 0));
    final base = (b * channels + c) * h * w;
    if (pool) {
      final left = (x0 + (dx * rw / ow).floor()).toInt().clamp(0, w),
          right = (x0 + ((dx + 1) * rw / ow).ceil()).toInt().clamp(0, w);
      final top = (y0 + (dy * rh / oh).floor()).toInt().clamp(0, h),
          bottom = (y0 + ((dy + 1) * rh / oh).ceil()).toInt().clamp(0, h);
      if (left >= right || top >= bottom) return 0;
      var peak = double.negativeInfinity;
      for (var y = top; y < bottom; y++)
        for (var xx = left; xx < right; xx++)
          peak = math.max(peak, x.values[base + y * w + xx].toDouble());
      return peak;
    }
    final ratio = n.integer('sampling_ratio', 0),
        sx = ratio > 0 ? ratio : (rw / ow).ceil(),
        sy = ratio > 0 ? ratio : (rh / oh).ceil(),
        maxMode = n.attrs['mode'] == 'max';
    var acc = maxMode ? double.negativeInfinity : 0.0;
    for (var y = 0; y < sy; y++)
      for (var xx = 0; xx < sx; xx++) {
        final py = y0 + (dy + (y + 0.5) / sy) * rh / oh,
            px = x0 + (dx + (xx + 0.5) / sx) * rw / ow;
        var val = 0.0;
        if (px >= -1 && px <= w && py >= -1 && py <= h) {
          final yt = _taps(py, 'linear', h, (c) => c.clamp(0, h - 1)),
              xt = _taps(px, 'linear', w, (c) => c.clamp(0, w - 1));
          if (maxMode) {
            val = double.negativeInfinity;
            for (final (iy, ay) in yt)
              for (final (ix, ax) in xt)
                val = math.max(val, ay * ax * x.values[base + iy * w + ix]);
          } else {
            val = _sample(x, base, [w, 1], [yt, xt]);
          }
        }
        acc = maxMode ? math.max(val, acc) : val + acc;
      }
    return maxMode ? acc : acc / math.max(1, sx * sy);
  });
}
