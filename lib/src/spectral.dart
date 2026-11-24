part of '../flutter_onnx.dart';

Tensor _einsum(_Node n, List<Tensor> xs) {
  final equation =
          (n.attrs['equation'] as String).replaceAll(' ', '').split('->'),
      terms = equation.first.split(',');
  _require(terms.length == xs.length, 'Einsum term count mismatch');
  final counts = <String, int>{}, dims = <String, int>{};
  var ell = 0;
  for (var k = 0; k < terms.length; k++) {
    final letters = terms[k].replaceAll('...', '');
    final missing = xs[k].shape.length - letters.length;
    _require(missing >= 0, 'Einsum rank mismatch');
    if (terms[k].contains('...')) ell = math.max(ell, missing);
    for (final c in letters.split('')) counts[c] = (counts[c] ?? 0) + 1;
  }
  final implicit = [for (var j = 0; j < ell; j++) '@$j'];
  List<String> labels(String term, int count) {
    final parts = term.split('...');
    return [
      if (parts.first.isNotEmpty) ...parts.first.split(''),
      if (parts.length == 2) ...[
        ...implicit.skip(ell - count),
        if (parts.last.isNotEmpty) ...parts.last.split('')
      ]
    ];
  }

  final ins = <List<String>>[];
  for (var t = 0; t < terms.length; t++) {
    final shape = xs[t].shape,
        l = labels(
            terms[t], shape.length - terms[t].replaceAll('...', '').length);
    _require(l.length == shape.length, 'Einsum operand rank mismatch');
    ins.add(l);
    for (var j = 0; j < l.length; j++) {
      final old = dims[l[j]] ?? 1;
      _require(old == 1 || shape[j] == 1 || shape[j] == old,
          'Einsum dimension mismatch');
      dims[l[j]] = old == 1 ? shape[j] : old;
    }
  }
  final out = equation.length == 2
      ? labels(equation[1], ell)
      : [
          ...implicit,
          ...(counts.keys.where((k) => counts[k] == 1).toList()..sort())
        ];
  final reduce = dims.keys.where((k) => !out.contains(k)).toList(),
      all = [...out, ...reduce],
      shape = [for (final k in out) dims[k]!];
  final rshape = [for (final k in reduce) dims[k]!],
      rsize = _size(rshape),
      os = _strides(shape),
      rs = _strides(rshape);
  final coords = List.filled(all.length, 0),
      positions = [
        for (final l in ins) [for (final c in l) all.indexOf(c)]
      ],
      strides = [for (final x in xs) _strides(x.shape)];
  return _generate(xs.first.type, shape, (i) {
    for (var j = 0; j < shape.length; j++) coords[j] = i ~/ os[j] % shape[j];
    var sum = 0.0;
    for (var r = 0; r < rsize; r++) {
      for (var j = 0; j < reduce.length; j++)
        coords[out.length + j] = r ~/ rs[j] % rshape[j];
      var prod = 1.0;
      for (var t = 0; t < xs.length; t++) {
        var off = 0;
        for (var j = 0; j < ins[t].length; j++)
          if (xs[t].shape[j] != 1)
            off += coords[positions[t][j]] * strides[t][j];
        prod *= xs[t].values[off];
      }
      sum += prod;
    }
    return sum;
  });
}

Tensor _spectral(_Node n, List<Tensor?> args) {
  final x = args[0]!;
  if (n.op == 'MelWeightMatrix') {
    final mel = x.values.single.toInt(),
        fft = args[1]!.values.single.toInt(),
        bins = fft ~/ 2 + 1,
        rate = args[2]!.values.single;
    double hzMel(num f) => 2595 * math.log(1 + f / 700) / math.ln10;
    final lo = hzMel(args[3]!.values.single),
        hi = hzMel(args[4]!.values.single);
    final points = List.generate(
        mel + 2,
        (i) => ((fft + 1) *
                700 *
                (math.pow(10, (lo + i * (hi - lo) / (mel + 2)) / 2595) - 1) /
                rate)
            .floor());
    return _generate(_type(n.integer('output_datatype', 1)), [bins, mel], (i) {
      final f = i ~/ mel,
          j = i % mel,
          a = points[j],
          b = points[j + 1],
          c = points[j + 2];
      if (f == b) return 1;
      if (f <= a || f >= c) return 0;
      return f < b ? (f - a) / (b - a) : (c - f) / (c - b);
    });
  }
  if (n.op == 'STFT') {
    final step = args[1]!.values.single.toInt(),
        window = args.elementAtOrNull(2),
        length =
            args.elementAtOrNull(3)?.values.single.toInt() ?? window!.length;
    final samples = x.shape[1],
        components = x.shape.length == 3 ? x.shape[2] : 1,
        frames = 1 + (samples - length) ~/ step;
    final freq = n.integer('onesided', 1) == 0 ? length : length ~/ 2 + 1;
    return _generate(x.type, [x.shape[0], frames, freq, 2], (i) {
      final part = i % 2,
          k = i ~/ 2 % freq,
          frame = i ~/ (2 * freq) % frames,
          batch = i ~/ (2 * freq * frames);
      var sum = 0.0;
      for (var j = 0; j < length; j++) {
        final base = ((batch * samples + frame * step + j) * components),
            re = x.values[base],
            im = components == 2 ? x.values[base + 1] : 0;
        final angle = -2 * math.pi * j * k / length;
        sum += (window?.values[j] ?? 1) *
            (part == 0
                ? re * math.cos(angle) - im * math.sin(angle)
                : re * math.sin(angle) + im * math.cos(angle));
      }
      return sum;
    });
  }
  final rank = x.shape.length,
      axis = _axis(
          args.elementAtOrNull(2)?.values.single.toInt() ??
              n.integer('axis', -2),
          rank);
  final inv = n.integer('inverse', 0) != 0,
      one = n.integer('onesided', 0) != 0,
      realInverse = inv && one;
  final size = args.elementAtOrNull(1)?.values.single.toInt() ??
          (realInverse ? 2 * (x.shape[axis] - 1) : x.shape[axis]),
      comp = x.shape.last;
  final freq = one && !inv ? size ~/ 2 + 1 : size, parts = realInverse ? 1 : 2;
  final dims = [...x.shape]
    ..[axis] = freq
    ..[rank - 1] = parts;
  _require(axis < rank - 1 && size > 0 && (comp == 1 || comp == 2),
      'Invalid DFT dimensions');
  final stride = _size(x.shape.sublist(axis + 1, rank - 1));
  final outer = _size(x.shape.take(axis)), output = Float64List(_size(dims));
  final scratch = Float32List(1);
  double round(double v) {
    if (x.type == TensorType.float64) return v;
    scratch[0] = v;
    return scratch[0];
  }

  for (var batch = 0; batch < outer; batch++)
    for (var lane = 0; lane < stride; lane++) {
      final signal = List.generate(size, (j) {
        final mirror = realInverse && j > size ~/ 2, t = mirror ? size - j : j;
        if (t >= x.shape[axis]) return (0.0, 0.0);
        final k = ((batch * x.shape[axis] + t) * stride + lane) * comp;
        return (
          x.values[k].toDouble(),
          comp == 2 ? (mirror ? -1 : 1) * x.values[k + 1].toDouble() : 0.0
        );
      });
      final transformed = _fourier(signal, inv ? 1 : -1, round);
      for (var k = 0; k < freq; k++) {
        final dest = ((batch * freq + k) * stride + lane) * parts;
        final v = transformed[k], scale = inv ? round(1 / size) : 1.0;
        output[dest] = v.$1 * scale;
        if (parts == 2)
          output[dest + 1] =
              comp == 1 && (k == 0 || 2 * k == size) ? 0 : v.$2 * scale;
      }
    }
  return Tensor.fromValues(x.type, dims, output);
}

// Decimation in frequency. The half-length transforms produce even and odd
// frequency bins. Pairing conjugate roots in the odd-length base transform
// avoids a separate trigonometric evaluation for the negative frequencies.
List<(double, double)> _fourier(
    List<(double, double)> x, int sign, double Function(double) round) {
  final n = x.length;
  if (n <= 1) return x;
  if (n.isEven) {
    final h = n ~/ 2, even = <(double, double)>[], odd = <(double, double)>[];
    for (var j = 0; j < h; j++) {
      final a = x[j],
          b = x[j + h],
          re = round(a.$1 - b.$1),
          im = round(a.$2 - b.$2);
      final angle = 2 * math.pi * j / n,
          c = round(math.cos(angle)),
          s = sign * round(math.sin(angle));
      even.add((round(a.$1 + b.$1), round(a.$2 + b.$2)));
      odd.add((
        round(round(re * c) - round(im * s)),
        round(round(re * s) + round(im * c))
      ));
    }
    final e = _fourier(even, sign, round), o = _fourier(odd, sign, round);
    return [
      for (var k = 0; k < h; k++) ...[e[k], o[k]]
    ];
  }
  return List.generate(n, (k) {
    var re = x.first.$1, im = x.first.$2, crossRe = 0.0, crossIm = 0.0;
    for (var j = 1; j <= n ~/ 2; j++) {
      final a = x[j], b = x[n - j], angle = 2 * math.pi * (k * j % n) / n;
      final c = round(math.cos(angle)), s = sign * round(math.sin(angle));
      re = round(re + round(round(a.$1 + b.$1) * c));
      im = round(im + round(round(a.$2 + b.$2) * c));
      crossIm = round(crossIm + round(round(a.$1 - b.$1) * s));
      crossRe = round(crossRe + round(round(a.$2 - b.$2) * s));
    }
    return (round(re - crossRe), round(im + crossIm));
  });
}
