part of '../flutter_onnx.dart';

sealed class OnnxValue {}

class OnnxSequence implements OnnxValue {
  final List<OnnxValue> elements;
  OnnxSequence(Iterable<OnnxValue> values)
      : elements = List.unmodifiable(values);
}

class OnnxOptional implements OnnxValue {
  final OnnxValue? value;
  const OnnxOptional(this.value);
}

Tensor _takeAxis(Tensor t, int axis, int index) {
  final r = _slice(t, [index], [index + 1], Tensor.int64([1], [axis]), null);
  return r._reshape([...r.shape]..removeAt(axis));
}

Tensor _stack(List<Tensor> tensors, int axis) {
  _require(tensors.isNotEmpty, 'Cannot infer shape of an empty stack');
  return _concat([
    for (final t in tensors) t._reshape([...t.shape]..insert(axis, 1))
  ], axis);
}

List<OnnxValue> _valueNode(
    _Node n, List<OnnxValue?> args, Map<String, OnnxValue> outer) {
  int index(int arg, int fallback, int length, {bool insert = false}) {
    final t = args.elementAtOrNull(arg) as Tensor?;
    var p = t?.values.single.toInt() ?? fallback;
    if (p < 0) p += length;
    _require(p >= 0 && p < length + (insert ? 1 : 0),
        'Sequence index out of bounds');
    return p;
  }

  List<OnnxValue> invoke(OnnxSession graph, List<OnnxValue> values) {
    _require(
        values.length == graph.inputs.length, 'Subgraph input count mismatch');
    return graph
        ._evaluate(Map.fromIterables(graph.inputs.keys, values), outer)
        .values
        .toList();
  }

  switch (n.op) {
    case 'Identity':
      return [args.first!];
    case 'SequenceEmpty':
      return [OnnxSequence([])];
    case 'SequenceConstruct':
      return [OnnxSequence(args.cast<OnnxValue>())];
    case 'SequenceLength':
      return [
        Tensor.int64([], [(args[0] as OnnxSequence).elements.length])
      ];
    case 'SequenceAt':
      final seq = (args[0] as OnnxSequence).elements;
      return [seq[index(1, 0, seq.length)]];
    case 'SequenceErase':
      final seq = [...(args[0] as OnnxSequence).elements];
      seq.removeAt(index(1, seq.length - 1, seq.length));
      return [OnnxSequence(seq)];
    case 'SequenceInsert':
      final seq = [...(args[0] as OnnxSequence).elements];
      seq.insert(index(2, seq.length, seq.length, insert: true), args[1]!);
      return [OnnxSequence(seq)];
    case 'Optional':
      return [OnnxOptional(args.firstOrNull)];
    case 'OptionalGetElement':
      final opt = args.first;
      if (opt is OnnxOptional) {
        if (opt.value == null) throw StateError('Empty optional');
        return [opt.value!];
      }
      return [opt!];
    case 'OptionalHasElement':
      final opt = args.firstOrNull;
      return [
        Tensor.fromValues(TensorType.boolean, [],
            [opt == null || (opt is OnnxOptional && opt.value == null) ? 0 : 1])
      ];
    case 'ConcatFromSequence':
      final ts = (args[0] as OnnxSequence).elements.cast<Tensor>();
      final stack = n.integer('new_axis', 0) != 0,
          ax = _axis(
              n.integer('axis', 0), ts.first.shape.length + (stack ? 1 : 0));
      return [stack ? _stack(ts, ax) : _concat(ts, ax)];
    case 'SplitToSequence':
      final x = args[0] as Tensor,
          ax = _axis(n.integer('axis', 0), x.shape.length),
          dim = x.shape[ax];
      final split = args.elementAtOrNull(1) as Tensor?, vals = <Tensor>[];
      if (split == null) {
        for (var i = 0; i < dim; i++) {
          final sliced = _takeAxis(x, ax, i);
          vals.add(n.integer('keepdims', 1) == 0
              ? sliced
              : sliced._reshape([...sliced.shape]..insert(ax, 1)));
        }
      } else {
        final size = split.values.firstOrNull?.toInt() ?? 0;
        if (split.shape.isEmpty)
          _require(size > 0, 'Split size must be positive');
        final sizes = split.shape.isEmpty
            ? [for (var i = 0; i < dim; i += size) math.min(dim - i, size)]
            : _indices(split);
        _require(sizes.fold(0, (a, b) => a + b) == dim,
            'Split sizes differ from dimension');
        var pos = 0;
        for (final s in sizes) {
          vals.add(_slice(x, [pos], [pos += s], Tensor.int64([1], [ax]), null));
        }
      }
      return [OnnxSequence(vals)];
    case 'If':
      final condition = (args.first as Tensor).values.single != 0;
      return invoke(
          n.attrs[condition ? 'then_branch' : 'else_branch'] as OnnxSession,
          []);
    case 'SequenceMap':
      final body = n.attrs['body'] as OnnxSession,
          seq = args.first as OnnxSequence;
      final outputs = [for (final _ in body.outputs.keys) <OnnxValue>[]];
      for (var i = 0; i < seq.elements.length; i++) {
        final values = invoke(body, [
          for (final arg in args) arg is OnnxSequence ? arg.elements[i] : arg!
        ]);
        for (var j = 0; j < values.length; j++) outputs[j].add(values[j]);
      }
      return [for (final out in outputs) OnnxSequence(out)];
    case 'Loop':
      final body = n.attrs['body'] as OnnxSession;
      final max = (args.elementAtOrNull(0) as Tensor?)?.values.single.toInt();
      var cond = (args.elementAtOrNull(1) as Tensor?)?.values.single != 0;
      var state = args.skip(2).cast<OnnxValue>().toList();
      final scans = [
        for (var i = state.length; i < n.outputs.length; i++) <Tensor>[]
      ];
      for (var i = 0; cond && (max == null || i < max); i++) {
        final values = invoke(body, [
          Tensor.int64([], [i]),
          Tensor.fromValues(TensorType.boolean, [], [cond ? 1 : 0]),
          ...state
        ]);
        cond = (values.first as Tensor).values.single != 0;
        state = values.sublist(1, state.length + 1);
        for (var j = 0; j < scans.length; j++) {
          scans[j].add(values[1 + state.length + j] as Tensor);
        }
      }
      return [
        ...state,
        for (var j = 0; j < scans.length; j++)
          _scanResult(
              scans[j], body.outputs.values.elementAt(1 + state.length + j), 0)
      ];
    case 'Scan':
      final body = n.attrs['body'] as OnnxSession,
          count = n.integer('num_scan_inputs', 1);
      final scans = args.skip(args.length - count).cast<Tensor>().toList();
      var state = args.take(args.length - count).cast<OnnxValue>().toList();
      final ia = n.ints('scan_input_axes') ?? List.filled(count, 0),
          dir = n.ints('scan_input_directions') ?? List.filled(count, 0);
      final axes = [
            for (var i = 0; i < count; i++) _axis(ia[i], scans[i].shape.length)
          ],
          steps = scans.first.shape[axes.first];
      final streams = [
        for (var k = state.length; k < n.outputs.length; k++) <Tensor>[]
      ];
      for (var t = 0; t < steps; t++) {
        final values = invoke(body, [
          ...state,
          for (var i = 0; i < count; i++)
            _takeAxis(scans[i], axes[i], dir[i] == 0 ? t : steps - 1 - t)
        ]);
        state = values.take(state.length).toList();
        for (var j = 0; j < streams.length; j++)
          streams[j].add(values[state.length + j] as Tensor);
      }
      final oa = n.ints('scan_output_axes') ?? List.filled(streams.length, 0),
          od = n.ints('scan_output_directions') ??
              List.filled(streams.length, 0);
      return [
        ...state,
        for (var j = 0; j < streams.length; j++)
          _scanResult(od[j] == 0 ? streams[j] : streams[j].reversed.toList(),
              body.outputs.values.elementAt(state.length + j), oa[j])
      ];
    default:
      return _executeNode(n, args.cast<Tensor?>());
  }
}

Tensor _scanResult(List<Tensor> tensors, TensorInfo info, int axis) {
  if (tensors.isNotEmpty)
    return _stack(tensors, _axis(axis, tensors.first.shape.length + 1));
  if (info.dimensions == null || info.dimensions!.any((d) => d is! int))
    throw UnsupportedError('Cannot infer empty scan output shape');
  final dims = info.dimensions!.cast<int>().toList();
  dims.insert(_axis(axis, dims.length + 1), 0);
  return Tensor.fromValues(info.type, dims, []);
}
