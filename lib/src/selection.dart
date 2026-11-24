part of '../flutter_onnx.dart';

List<Tensor> _loss(_Node n, List<Tensor?> args) {
  final x = args[0]!, targets = args[1]!, weights = args.elementAtOrNull(2);
  final logp = n.op == 'SoftmaxCrossEntropyLoss' ? _logSoftmax(x, 1) : x;
  final channels = x.shape[1],
      dim = _size(x.shape.skip(2)),
      ignore = n.attrs['ignore_index'];
  final vals = Float64List(targets.length);
  var sum = 0.0, total = 0.0;
  for (var i = 0; i < vals.length; i++) {
    final label = targets.values[i].toInt();
    if (label == ignore) continue;
    _require(label >= 0 && label < channels, 'Invalid class index');
    final weight = weights?.values[label].toDouble() ?? 1.0;
    vals[i] =
        -weight * logp.values[(i ~/ dim * channels + label) * dim + i % dim];
    sum += vals[i];
    total += weight;
  }
  final reduction = n.attrs['reduction'] ?? 'mean';
  return [
    reduction == 'none'
        ? Tensor.fromValues(x.type, targets.shape, vals)
        : Tensor.fromValues(
            x.type, [], [reduction == 'sum' ? sum : sum / total]),
    if (n.outputs.length > 1) logp
  ];
}

Tensor _nms(_Node n, List<Tensor?> args) {
  final boxes = args[0]!,
      scores = args[1]!,
      cap = args.elementAtOrNull(2)?.values.single.toInt() ?? 0;
  final threshold = args.elementAtOrNull(3)?.values.single ?? 0,
      minimum =
          args.elementAtOrNull(4)?.values.single ?? double.negativeInfinity;
  final [b, classes, count] = scores.shape;
  final selected = <int>[];
  for (var batch = 0; batch < b; batch++) {
    List<double> rect(int i) {
      final p = (batch * count + i) * 4;
      final a = boxes.values[p].toDouble(),
          b = boxes.values[p + 1].toDouble(),
          c = boxes.values[p + 2].toDouble(),
          d = boxes.values[p + 3].toDouble();
      if (n.integer('center_point_box', 0) == 1)
        return [a - c / 2, b - d / 2, a + c / 2, b + d / 2];
      return [math.min(a, c), math.min(b, d), math.max(a, c), math.max(b, d)];
    }

    final rects = List.generate(count, rect);
    for (var c = 0; c < classes; c++) {
      num score(int idx) => scores.values[(batch * classes + c) * count + idx];
      final order = [
        for (var j = 0; j < count; j++)
          if (score(j) > minimum) j
      ]..sort((a, b) {
          final v = score(b).compareTo(score(a));
          return v == 0 ? a - b : v;
        });
      final kept = <int>[];
      for (final j in order) {
        if (kept.length >= cap) break;
        final r = rects[j];
        bool overlap(int idx) {
          final s = rects[idx],
              w = math.max(0, math.min(r[2], s[2]) - math.max(r[0], s[0])),
              h = math.max(0, math.min(r[3], s[3]) - math.max(r[1], s[1]));
          final area = w * h,
              union = (r[2] - r[0]) * (r[3] - r[1]) +
                  (s[2] - s[0]) * (s[3] - s[1]) -
                  area;
          return union > 0 && area / union > threshold;
        }

        if (!kept.any(overlap)) {
          kept.add(j);
          selected.addAll([batch, c, j]);
        }
      }
    }
  }
  return Tensor.int64([selected.length ~/ 3, 3], selected);
}

List<Tensor> _unique(_Node n, Tensor x) {
  final flat = !n.attrs.containsKey('axis'),
      shape = flat ? [x.length] : x.shape,
      axis = flat ? 0 : _axis(n.integer('axis', 0), shape.length);
  final width = shape[axis],
      inner = _size(shape.skip(axis + 1)),
      outer = _size(shape.take(axis));
  final slices = [
    for (var i = 0; i < width; i++)
      [
        for (var g = 0; g < outer; g++)
          for (var j = 0; j < inner; j++) x.values[(g * width + i) * inner + j]
      ]
  ];
  int compare(int a, int b) {
    for (var j = 0; j < slices[a].length; j++) {
      final d = slices[a][j].compareTo(slices[b][j]);
      if (d != 0) return d;
    }
    return 0;
  }

  final order = List.generate(width, (i) => i)
    ..sort((a, b) {
      final c = compare(a, b);
      return c == 0 ? a - b : c;
    });
  final indices = <int>[];
  for (final i in order) {
    if (indices.isEmpty || compare(i, indices.last) != 0) indices.add(i);
  }
  if (n.integer('sorted', 1) == 0) indices.sort();
  final inverse = List.generate(
          width, (i) => indices.indexWhere((p) => compare(i, p) == 0)),
      counts = List.filled(indices.length, 0);
  for (final j in inverse) counts[j]++;
  final dims = [...shape]..[axis] = indices.length;
  return [
    _reorder(
        x,
        dims,
        (i) =>
            (i ~/ (indices.length * inner) * width +
                    indices[i ~/ inner % indices.length]) *
                inner +
            i % inner),
    Tensor.int64([indices.length], indices),
    Tensor.int64([width], inverse),
    Tensor.int64([counts.length], counts)
  ];
}

List<Tensor> _stringOp(_Node n, List<Tensor?> args) {
  final x = args.first!;
  if (n.op == 'StringConcat' || n.op == 'Equal') {
    final b = args[1]!,
        dims = _broadcastShape(x.shape, b.shape),
        aMap = _IndexMap(x.shape, dims),
        bMap = _IndexMap(b.shape, dims);
    if (n.op == 'Equal')
      return [
        _generate(
            TensorType.boolean,
            dims,
            (i) => x.stringValues[aMap.at(i)] == b.stringValues[bMap.at(i)]
                ? 1
                : 0)
      ];
    return [
      Tensor.strings(
          dims,
          List.generate(_size(dims),
              (i) => x.stringValues[aMap.at(i)] + b.stringValues[bMap.at(i)]))
    ];
  }
  if (n.op == 'RegexFullMatch') {
    final expr = RegExp('^(?:${n.attrs['pattern']})\u0024', unicode: true);
    return [
      _generate(TensorType.boolean, x.shape,
          (i) => expr.hasMatch(x.stringValues[i]) ? 1 : 0)
    ];
  }
  if (n.op == 'StringNormalizer') {
    final sensitive = n.integer('is_case_sensitive', 0) != 0;
    String key(String s) => sensitive ? s : s.toLowerCase();
    final stops = {
      for (final s in n.attrs['stopwords'] as List<String>? ?? <String>[])
        key(s)
    };
    final action = n.attrs['case_change_action'] ?? 'NONE';
    final strings = [
      for (final s in x.stringValues)
        if (!stops.contains(key(s)))
          action == 'LOWER'
              ? s.toLowerCase()
              : action == 'UPPER'
                  ? s.toUpperCase()
                  : s
    ];
    if (strings.isEmpty) strings.add('');
    return [
      Tensor.strings([if (x.shape.length == 2) 1, strings.length], strings)
    ];
  }
  final delimiter = n.attrs['delimiter'] as String?,
      limit = n.integer('maxsplit', -1);
  List<String> split(String s) {
    if (s.isEmpty && (delimiter == null || delimiter.isEmpty)) return [];
    final whitespace = delimiter == null || delimiter.isEmpty;
    final source = whitespace ? s.trim() : s;
    if (source.isEmpty && whitespace) return [];
    final matches =
        (whitespace ? RegExp(r'\s+') : RegExp(RegExp.escape(delimiter)))
            .allMatches(source);
    var pos = 0;
    final parts = <String>[];
    for (final match in matches) {
      if (limit >= 0 && parts.length >= limit) break;
      parts.add(source.substring(pos, match.start));
      pos = match.end;
    }
    parts.add(source.substring(pos));
    return parts;
  }

  final rows = x.stringValues.map(split).toList(),
      counts = [for (final row in rows) row.length];
  final width = counts.fold<int>(0, math.max);
  return [
    Tensor.strings([
      ...x.shape,
      width
    ], [
      for (final row in rows) ...[
        ...row,
        ...List.filled(width - row.length, '')
      ]
    ]),
    Tensor.int64(x.shape, counts)
  ];
}
