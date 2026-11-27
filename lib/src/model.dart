part of '../flutter_onnx.dart';

/// Tensor interface declared by an ONNX graph. Dimensions may be integers,
/// symbolic names or null (unknown). A null dimensions list means unknown rank.
class TensorInfo {
  final String name;
  final TensorType type;
  final List<Object?>? dimensions;
  final String kind;
  final TensorInfo? element;

  /// Declared key type of a map. Its value type is available in [element].
  final TensorType? keyType;
  TensorInfo._(this.name, this.type, this.dimensions,
      [this.kind = "tensor", this.element, this.keyType]);

  factory TensorInfo._read(_Proto value) {
    return TensorInfo._type(value.string(1), value.message(2));
  }
  factory TensorInfo._type(String name, _Proto tensor) {
    if (tensor.has(4) || tensor.has(9)) {
      final sequence = tensor.has(4);
      final inner =
          TensorInfo._type(name, tensor.message(sequence ? 4 : 9).message(1));
      return TensorInfo._(
          name, inner.type, null, sequence ? 'sequence' : 'optional', inner);
    }
    if (tensor.has(5)) {
      final map = tensor.message(5);
      final keyType = _type(map.integer(1));
      if (!_mapKeyTypes.contains(keyType))
        throw FormatException('Invalid map key type $keyType');
      final element = TensorInfo._type(name, map.message(2));
      return TensorInfo._(name, element.type, null, 'map', element, keyType);
    }
    if (!tensor.has(1)) throw UnsupportedError('Expected dense tensor type');
    final t = tensor.message(1);
    final dims = t.has(2)
        ? List<Object?>.unmodifiable(
            t.message(2).messages(1).map((d) => d.has(1)
                ? d.integer(1)
                : d.has(2)
                    ? d.string(2)
                    : null))
        : null;
    return TensorInfo._(name, _type(t.integer(1)), dims);
  }

  void _check(OnnxValue value, Map<String, int> symbols,
      {bool checkDimensions = true}) {
    if (kind == 'optional') {
      final present = value is OnnxOptional ? value.value : value;
      if (present != null)
        element!._check(present, symbols, checkDimensions: checkDimensions);
      return;
    }
    if (kind == 'map') {
      if (value is! OnnxMap || value.keyType != keyType) {
        throw ArgumentError('$name expects a map with $keyType keys');
      }
      for (final item in value.entries.values) {
        element!._check(item, symbols, checkDimensions: checkDimensions);
      }
      return;
    }
    if (kind == 'sequence') {
      if (value is! OnnxSequence)
        throw ArgumentError('$name expects a sequence');
      for (final item in value.elements) {
        // Sequence kernels, especially Loop, can append differently shaped
        // tensors. Preserve element shape annotations as metadata while
        // checking the runtime element kind and dtype.
        element!._check(item, symbols, checkDimensions: false);
      }
      return;
    }
    if (value is! Tensor) throw ArgumentError('$name expects a tensor');
    final t = value;
    _require(type == t.type, '$name expects $type, got ${t.type}');
    final dims = dimensions;
    if (!checkDimensions || dims == null) return;
    _require(
        dims.length == t.shape.length, '$name expects rank ${dims.length}');
    for (var i = 0; i < dims.length; i++) {
      final d = dims[i];
      final expected =
          d is String ? symbols.putIfAbsent(d, () => t.shape[i]) : d;
      _require(expected == null || expected == t.shape[i],
          '$name axis $i expects $expected, got ${t.shape[i]}');
    }
  }
}

/// A reusable execution plan. Load once, then call [run] with named tensors.
class OnnxSession {
  final int opset;
  final Map<String, TensorInfo> inputs;
  final Map<String, TensorInfo> outputs;
  final Map<String, Tensor> _weights;
  final List<_Node> _nodes;
  final Map<String, int> _uses;
  late final List<_Node> _pointwiseNodes =
      _fusePointwiseGelu(_nodes, outputs, _uses);
  late final List<_Node> _channelwiseNodes = _fuseScaledResidual(
      _fuseChannelNorm(_pointwiseNodes, outputs, _uses),
      _weights,
      inputs,
      outputs,
      _uses);
  late final List<_Node> _pairedNodes =
      _fusePointwisePairs(_channelwiseNodes, _weights, inputs, outputs, _uses);
  final Set<String> _captures;
  OnnxSession._(this.opset, this.inputs, this.outputs, this._weights,
      this._nodes, this._uses, this._captures);

  /// Supply external weight files by their exact ONNX location.
  factory OnnxSession.fromBytes(Uint8List bytes,
      {Map<String, Uint8List> externalData = const {}}) {
    try {
      final model = _Proto(bytes);
      final ir = model.integer(1);
      if (ir < 3 || ir > 14)
        throw UnsupportedError('ONNX IR $ir (supported: 3–14)');
      final imports =
          model.messages(8).where((m) => m.string(1).isEmpty).toList();
      if (imports.length != 1)
        throw const FormatException('Expected one default-domain opset import');
      final version = imports.single.integer(2);
      if (version < 13 || version > 28)
        throw UnsupportedError('ONNX opset $version (supported: 13–28)');
      final functions = _FunctionScope.fromModel(model);
      return OnnxSession._readGraph(model.message(7), version,
          externalData: externalData, functions: functions);
    } on TypeError catch (e) {
      throw FormatException('Invalid ONNX protobuf field: $e');
    }
  }
  factory OnnxSession._readGraph(_Proto graph, int version,
      {bool allowCapture = false,
      _FunctionScope functions = const _FunctionScope(),
      Map<String, Uint8List> externalData = const {}}) {
    final captures = <String>{};

    Map<String, TensorInfo> interface(int field) {
      final result = <String, TensorInfo>{};
      for (final value in graph.messages(field)) {
        final info = TensorInfo._read(value);
        if (info.name.isEmpty || result.containsKey(info.name))
          throw FormatException('Invalid graph value "${info.name}"');
        result[info.name] = info;
      }
      return Map.unmodifiable(result);
    }

    final inputs = interface(11);
    final outputs = interface(12);
    final nodes = graph
        .messages(1)
        .map((p) => _Node.read(p, version, externalData, functions))
        .toList();
    final weights = <String, Tensor>{};
    void addWeight(String name, Tensor tensor) {
      if (name.isEmpty) throw const FormatException('Unnamed initializer');
      final prior = weights[name];
      if (prior == null) {
        weights[name] = tensor;
        return;
      }
      // Some exporters serialize the same named weight at different offsets.
      if (prior.type != tensor.type || !_sameShape(prior.shape, tensor.shape))
        throw FormatException('Conflicting initializer $name');
      final List<Object> a = prior.type == TensorType.string
          ? prior.stringValues
          : prior.toBytes();
      final List<Object> b = tensor.type == TensorType.string
          ? tensor.stringValues
          : tensor.toBytes();
      for (var i = 0; i < a.length; i++)
        if (a[i] != b[i])
          throw FormatException('Conflicting initializer $name');
    }

    for (final t in graph.messages(5))
      addWeight(t.string(8), _tensor(t, externalData: externalData));
    for (final t in graph.messages(15))
      addWeight(t.message(1).string(8), _sparseTensor(t, externalData));
    final known = {...inputs.keys, ...weights.keys};
    final uses = <String, int>{};
    for (final node in nodes) {
      for (final name in node.references) {
        if (!known.contains(name)) {
          if (!allowCapture)
            throw FormatException('${node.label}: unknown input $name');
          captures.add(name);
          known.add(name);
        }
        uses.update(name, (v) => v + 1, ifAbsent: () => 1);
      }
      for (final name in node.outputs.where((v) => v.isNotEmpty)) {
        if (!known.add(name))
          throw FormatException('Duplicate graph value $name');
      }
    }
    for (final name in outputs.keys) {
      if (!known.contains(name)) {
        if (!allowCapture) throw FormatException('Unknown graph output $name');
        captures.add(name);
      }
    }
    final optimized =
        _fuseTransposedWeights(nodes, weights, inputs, outputs, uses);
    final planned = _fuseGelu(optimized, weights, inputs, outputs, uses);
    return OnnxSession._(
        version, inputs, outputs, weights, planned, uses, captures);
  }

  Map<String, Tensor> run(Map<String, Tensor> feeds) =>
      runValues(feeds).cast<String, Tensor>();

  /// Execute a graph containing tensor, sequence or optional values.
  Map<String, OnnxValue> runValues(Map<String, OnnxValue> feeds) =>
      _evaluate(feeds, {});
  Map<String, OnnxValue> _evaluate(
      Map<String, OnnxValue> feeds, Map<String, OnnxValue> outer) {
    final frame = _Frame(this, feeds, outer);
    for (final n in _nodes) {
      frame.store(n, _valueNode(n, frame.args(n), frame.values));
    }
    return frame.results();
  }

  /// Offloads supported matrix products and executes other nodes locally.
  /// [onAccelerated] is called only after a backend produces a result.
  Future<Map<String, Tensor>> runAsync(Map<String, Tensor> feeds,
      {MatMulBackend? backend,
      void Function(String node)? onAccelerated}) async {
    if (backend == null) return run(feeds);
    final frame = _Frame(this, feeds, {});
    final plan =
        backend is PointwisePairBackend && backend.supportsPointwisePairs
            ? _pairedNodes
            : backend is ChannelwiseBackend
                ? _channelwiseNodes
                : backend is PointwiseConvBackend
                    ? _pointwiseNodes
                    : _nodes;
    for (final node in plan) {
      final args = frame.args(node);
      if (node.op == _pointwisePair) {
        final value = await _runPointwisePair(node, args.cast<Tensor?>(),
            backend as PointwisePairBackend, onAccelerated);
        frame.store(node, [value]);
        continue;
      }
      final product = (node.op == 'MatMul' ||
              node.op == 'Gemm' ||
              node.op == _rowProduct ||
              node.op == 'Conv' ||
              node.op == _scaledResidual ||
              node.op == _channelNorm ||
              node.op == _pointwiseGelu)
          ? await _tryAccelerate(node, args.cast<Tensor?>(), backend)
          : null;
      if (product != null) onAccelerated?.call(node.label);
      if (product != null) {
        frame.store(node, [product]);
      } else if (node.op == _scaledResidual) {
        final t = args.cast<Tensor>();
        frame.store(node, [_binary('Add', t[2], _binary('Mul', t[1], t[0]))]);
      } else if (node.op == _channelNorm) {
        final tensors = args.cast<Tensor?>();
        final nlc = _transpose(tensors[0]!, [0, 2, 1]);
        final normalized = _normalize(node, [nlc, ...tensors.skip(1)]).first;
        frame.store(node, [
          _transpose(normalized, [0, 2, 1])
        ]);
      } else if (node.op == _pointwiseGelu) {
        final conv =
            _Node('Conv', node.label, node.outputs, node.inputs, node.attrs);
        frame.store(node, [
          _expandedGeluTensor(_executeNode(conv, args.cast<Tensor?>()).single)
        ]);
      } else {
        frame.store(node, _valueNode(node, args, frame.values));
      }
    }
    return frame.results().cast<String, Tensor>();
  }
}

/// Optional dense FLOAT matrix multiplication primitive, with row-major buffers.
/// Return null if this device cannot perform the product.
abstract interface class MatMulBackend {
  String get name;
  Future<Float32List?> multiply(Float32List left, Float32List right,
      {required int rows, required int inner, required int columns});
  Future<void> close();
}

/// Optional native kernel for dense unit-stride pointwise NCL convolutions.
/// Weights and bias are immutable. The GELU form includes the float32 rounding
/// of the exported Div, Erf, Add and two Mul operations.
abstract interface class PointwiseConvBackend implements MatMulBackend {
  Future<Float32List?> pointwise(Tensor input, Tensor weights, Tensor? bias,
      {required bool gelu});
}

/// Optional native primitives for channel-first, one-dimensional models.
/// Implementations may decline a tensor or layout by returning null.
abstract interface class ChannelwiseBackend implements PointwiseConvBackend {
  Float32List? normalizeChannels(
      Tensor input, Tensor scale, Tensor? bias, double epsilon);
  Float32List? depthwiseValid(
      Tensor input, Tensor weights, Tensor? bias, int dilation);
  Float32List? scaledResidual(
      Tensor input, Tensor channelScale, Tensor residual);
}

/// Optional pair of unit-stride NCL pointwise convolutions with the exported
/// GELU between them. A backend can keep the intermediate in its native layout.
abstract interface class PointwisePairBackend implements ChannelwiseBackend {
  bool get supportsPointwisePairs;
  Future<Float32List?> pointwisePair(Tensor input, Tensor firstWeights,
      Tensor? firstBias, Tensor secondWeights, Tensor? secondBias);
}

const _pointwisePair = '#PointwisePair';

List<_Node> _fusePointwisePairs(
    List<_Node> nodes,
    Map<String, Tensor> weights,
    Map<String, TensorInfo> inputs,
    Map<String, TensorInfo> outputs,
    Map<String, int> uses) {
  final first = <String, _Node>{};
  final removed = <_Node>{}, replacements = <_Node, _Node>{};
  bool unit(_Node node) {
    if (node.function != null ||
        node.inputs.length < 2 ||
        node.inputs.length > 3 ||
        inputs.containsKey(node.inputs[1])) return false;
    final w = weights[node.inputs[1]];
    return w != null &&
        w.type == TensorType.float32 &&
        w.shape.length == 3 &&
        w.shape.last == 1 &&
        node.integer('group', 1) == 1 &&
        _sameShape(node.ints('kernel_shape') ?? [1], [1]) &&
        _sameShape(node.ints('strides') ?? [1], [1]) &&
        _sameShape(node.ints('dilations') ?? [1], [1]) &&
        _sameShape(node.ints('pads') ?? [0, 0], [0, 0]) &&
        (node.attrs['auto_pad'] ?? 'NOTSET') == 'NOTSET';
  }

  for (final node in nodes) {
    if (node.op == _pointwiseGelu &&
        node.outputs.length == 1 &&
        uses[node.outputs.single] == 1 &&
        !outputs.containsKey(node.outputs.single) &&
        unit(node)) {
      first[node.outputs.single] = node;
    } else if (node.op == 'Conv' && node.inputs.isNotEmpty && unit(node)) {
      final begin = first[node.inputs.first];
      if (begin == null) continue;
      removed.add(begin);
      replacements[node] = _Node(_pointwisePair, node.label, node.outputs, [
        begin.inputs[0],
        begin.inputs[1],
        begin.inputs.elementAtOrNull(2) ?? '',
        node.inputs[1],
        node.inputs.elementAtOrNull(2) ?? '',
      ], {
        'first_label': begin.label
      });
    }
  }
  return [
    for (final node in nodes)
      if (!removed.contains(node)) replacements[node] ?? node
  ];
}

Future<Tensor> _runPointwisePair(
    _Node node,
    List<Tensor?> args,
    PointwisePairBackend backend,
    void Function(String node)? onAccelerated) async {
  final input = args[0]!, w1 = args[1]!, w2 = args[3]!;
  final firstLabel = node.attrs['first_label'] as String;
  final data = await backend.pointwisePair(input, w1, args[2], w2, args[4]);
  if (data != null) {
    final shape = [input.shape[0], w2.shape[0], input.shape[2]];
    if (data.length != _size(shape)) {
      throw StateError(
          '${backend.name} returned an incorrect pointwise pair length');
    }
    onAccelerated?.call(firstLabel);
    onAccelerated?.call(node.label);
    return Tensor._(input.type, shape, data.asUnmodifiableView());
  }
  Future<Tensor> project(String label, List<Tensor?> inputs, bool gelu) async {
    final conv = _Node('Conv', label, [], [], {});
    final accelerated = await _accelerateConv1d(
        gelu ? _Node(_pointwiseGelu, label, [], [], {}) : conv,
        inputs,
        backend);
    if (accelerated != null) {
      onAccelerated?.call(label);
      return accelerated;
    }
    final result = _executeNode(conv, inputs).single;
    return gelu ? _expandedGeluTensor(result) : result;
  }

  final intermediate = await project(firstLabel, [input, w1, args[2]], true);
  return project(node.label, [intermediate, w2, args[4]], false);
}

const _scaledResidual = '#ScaledResidual';
List<_Node> _fuseScaledResidual(
    List<_Node> nodes,
    Map<String, Tensor> weights,
    Map<String, TensorInfo> inputs,
    Map<String, TensorInfo> outputs,
    Map<String, int> uses) {
  final products = <String, _Node>{};
  final remove = <_Node>{}, replace = <_Node, _Node>{};
  for (final n in nodes) {
    if (n.function != null) continue;
    if (n.op == 'Mul' &&
        n.outputs.length == 1 &&
        uses[n.outputs.single] == 1 &&
        !outputs.containsKey(n.outputs.single)) {
      if (inputs.containsKey(n.inputs.first)) continue;
      final scale = weights[n.inputs.first];
      if (scale != null &&
          scale.type == TensorType.float32 &&
          scale.shape.length == 3 &&
          scale.shape[0] == 1 &&
          scale.shape[2] == 1) products[n.outputs.single] = n;
    } else if (n.op == 'Add') {
      final product = products[n.inputs[1]];
      if (product == null) continue;
      remove.add(product);
      replace[n] = _Node(_scaledResidual, n.label, n.outputs,
          [product.inputs[1], product.inputs[0], n.inputs[0]], {});
    }
  }
  return [
    for (final n in nodes)
      if (!remove.contains(n)) replace[n] ?? n
  ];
}

const _channelNorm = '#ChannelNormalization';
List<_Node> _fuseChannelNorm(
    List<_Node> nodes, Map<String, TensorInfo> outputs, Map<String, int> uses) {
  final private = {
    for (final n in nodes)
      if (n.function == null &&
          n.outputs.length == 1 &&
          uses[n.outputs.single] == 1 &&
          !outputs.containsKey(n.outputs.single))
        n.outputs.single: n
  };
  bool swap(_Node? n) =>
      n != null &&
      n.function == null &&
      n.op == 'Transpose' &&
      _sameShape(n.ints('perm') ?? [], [0, 2, 1]);
  final removed = <_Node>{}, replacements = <_Node, _Node>{};
  for (final end in nodes) {
    if (!swap(end)) continue;
    final norm = private[end.inputs.single];
    if (norm == null ||
        norm.op != 'LayerNormalization' ||
        ![-1, 2].contains(norm.integer('axis', -1))) continue;
    final begin = private[norm.inputs.first];
    if (!swap(begin) || removed.contains(begin)) continue;
    // A transpose ending one normalization may also start the next one.
    // Leave overlapping matches to the original plan.
    if (replacements.containsKey(begin)) continue;
    removed.addAll([begin!, norm]);
    replacements[end] = _Node(_channelNorm, norm.label, end.outputs,
        [begin.inputs.single, ...norm.inputs.skip(1)], norm.attrs);
  }
  return [
    for (final n in nodes)
      if (!removed.contains(n)) replacements[n] ?? n
  ];
}

const _pointwiseGelu = '#PointwiseGelu';

List<_Node> _fusePointwiseGelu(
    List<_Node> nodes, Map<String, TensorInfo> outputs, Map<String, int> uses) {
  final convs = <String, _Node>{};
  final replacements = <_Node, _Node>{}, removed = <_Node>{};
  for (final n in nodes) {
    if (n.op == 'Conv' &&
        n.function == null &&
        n.outputs.length == 1 &&
        uses[n.outputs.single] == 1 &&
        !outputs.containsKey(n.outputs.single)) {
      convs[n.outputs.single] = n;
    } else if (n.op == _expandedGelu) {
      final conv = convs[n.inputs.single];
      if (conv != null) {
        replacements[n] = _Node(
            _pointwiseGelu, conv.label, n.outputs, conv.inputs, conv.attrs);
        removed.add(conv);
      }
    }
  }
  return [
    for (final n in nodes)
      if (!removed.contains(n)) replacements[n] ?? n
  ];
}

Future<Tensor?> _tryAccelerate(
    _Node node, List<Tensor?> args, MatMulBackend backend) async {
  if (node.op == _scaledResidual && backend is ChannelwiseBackend) {
    final x = args[0]!;
    final data = backend.scaledResidual(x, args[1]!, args[2]!);
    return data == null
        ? null
        : Tensor._(x.type, x.shape, data.asUnmodifiableView());
  }
  if (node.op == _channelNorm && backend is ChannelwiseBackend) {
    final x = args[0]!;
    final data = backend.normalizeChannels(
        x, args[1]!, args.elementAtOrNull(2), node.real('epsilon', 1e-5));
    return data == null
        ? null
        : Tensor._(x.type, x.shape, data.asUnmodifiableView());
  }
  if (node.op == 'Conv' || node.op == _pointwiseGelu) {
    return _accelerateConv1d(node, args, backend);
  }
  if (node.op == _rowProduct) {
    final input = args[0]!, weights = args[1]!;
    final dims = _rowProductShape(input, weights);
    final rows = _size(input.shape.take(input.shape.length - 1)),
        inner = input.shape.last,
        columns = weights.shape.first;
    if (rows == 0 || inner == 0 || columns == 0) return null;
    final rhs = _transpose(input._reshape([rows, inner]), [1, 0]);
    final data = await backend.multiply(
        weights.values as Float32List, rhs.values as Float32List,
        rows: columns, inner: inner, columns: rows);
    if (data == null) return null;
    if (data.length != rows * columns)
      throw StateError('${backend.name} returned an incorrect matrix length');
    return _transpose(
        Tensor._(
            TensorType.float32, [columns, rows], data.asUnmodifiableView()),
        [1, 0])._reshape(dims);
  }
  if (node.op != 'MatMul' && node.op != 'Gemm') return null;
  var a = args[0]!, b = args[1]!;
  if (node.op == 'MatMul' &&
      a.type == TensorType.float32 &&
      b.type == a.type &&
      a.shape.length >= 2 &&
      b.shape.length > 2 &&
      a.length != 0 &&
      b.length != 0) {
    final rows = a.shape[a.shape.length - 2],
        inner = a.shape.last,
        columns = b.shape.last;
    _require(
        inner == b.shape[b.shape.length - 2], 'Matrix inner dimensions differ');
    final aBatch = a.shape.sublist(0, a.shape.length - 2),
        bBatch = b.shape.sublist(0, b.shape.length - 2);
    final batch = _broadcastShape(aBatch, bBatch),
        ma = _IndexMap(aBatch, batch),
        mb = _IndexMap(bBatch, batch);
    final shape = [...batch, rows, columns], out = Float32List(_size(shape));
    final av = a.values as Float32List, bv = b.values as Float32List;
    final count = _size(batch);
    for (var g = 0; g < count; g++) {
      final ap = ma.at(g) * rows * inner, bp = mb.at(g) * inner * columns;
      final values = await backend.multiply(
          Float32List.sublistView(av, ap, ap + rows * inner),
          Float32List.sublistView(bv, bp, bp + inner * columns),
          rows: rows,
          inner: inner,
          columns: columns);
      if (values == null) return null;
      if (values.length != rows * columns)
        throw StateError('${backend.name} returned an incorrect matrix length');
      out.setRange(g * rows * columns, (g + 1) * rows * columns, values);
    }
    return Tensor._(a.type, shape, out.asUnmodifiableView());
  }
  if (a.type != TensorType.float32 ||
      b.type != TensorType.float32 ||
      a.values.isEmpty ||
      b.values.isEmpty ||
      b.shape.length != 2 ||
      a.shape.length < 2) return null;
  List<int>? restoredShape;
  if (a.shape.length > 2) {
    if (node.op != 'MatMul') return null;
    restoredShape = [...a.shape.take(a.shape.length - 1), b.shape.last];
    a = a._reshape([a.values.length ~/ a.shape.last, a.shape.last]);
  }
  if (node.integer('transA', 0) != 0) a = _transpose(a, [1, 0]);
  if (node.integer('transB', 0) != 0) b = _transpose(b, [1, 0]);
  _require(a.shape[1] == b.shape[0], 'Matrix inner dimensions differ');
  final shape = [a.shape[0], b.shape[1]];
  final count = _size(shape);
  final bias = args.elementAtOrNull(2);
  if (bias != null && bias.type != TensorType.float32)
    throw UnsupportedError('Gemm requires FLOAT bias');
  final map = bias == null ? null : _IndexMap(bias.shape, shape);
  final data = await backend.multiply(
      a.values as Float32List, b.values as Float32List,
      rows: shape[0], inner: a.shape[1], columns: shape[1]);
  if (data == null) return null;
  if (data.length != count)
    throw StateError('${backend.name} returned an incorrect matrix length');
  final alpha = node.real('alpha', 1), beta = node.real('beta', 1);
  if (alpha == 1 && (bias == null || beta == 0))
    return Tensor._(
        TensorType.float32, restoredShape ?? shape, data.asUnmodifiableView());
  return _float(
      shape,
      (i) =>
          alpha * data[i] +
          (bias == null || beta == 0 ? 0 : beta * bias.values[map!.at(i)]));
}

class _Frame {
  final OnnxSession session;
  final symbols = <String, int>{};
  final Map<String, OnnxValue> values;
  final Map<String, int> remaining;
  _Frame(
      this.session, Map<String, OnnxValue> feeds, Map<String, OnnxValue> outer)
      : values = {
          for (final k in session._captures) k: outer[k]!,
          ...session._weights
        },
        remaining = {...session._uses} {
    for (final name in feeds.keys) {
      _require(session.inputs.containsKey(name), 'Unknown graph input $name');
    }
    for (final info in session.inputs.values) {
      final value = feeds[info.name] ??
          session._weights[info.name] ??
          (info.kind == "optional" ? const OnnxOptional(null) : null);
      if (value == null)
        throw ArgumentError('Missing graph input ${info.name}');
      info._check(value, symbols);
      values[info.name] = value;
    }
  }

  List<OnnxValue?> args(_Node n) => n.inputs.map((key) => values[key]).toList();
  void store(_Node n, List<OnnxValue> outputs) {
    for (final (i, key) in n.outputs.indexed) {
      if (key.isNotEmpty &&
          (remaining.containsKey(key) || session.outputs.containsKey(key)))
        values[key] = outputs[i];
    }
    for (final k in n.references) {
      if ((remaining[k] = remaining[k]! - 1) == 0 &&
          !session.outputs.containsKey(k)) values.remove(k);
    }
  }

  Map<String, OnnxValue> results() {
    final result = <String, OnnxValue>{};
    for (final info in session.outputs.values) {
      final t = values[info.name]!;
      info._check(t, symbols);
      result[info.name] = t;
    }
    return result;
  }
}

class _Node {
  final String op, label;
  final List<String> outputs;
  final List<String> inputs;
  final Map<String, Object> attrs;
  late final _OnnxRandom random = _OnnxRandom((attrs["seed"] as num?)?.toInt());
  final _FunctionCall? function;
  _Node(this.op, this.label, this.outputs, this.inputs, this.attrs,
      [this.function]);

  Iterable<String> get references sync* {
    yield* inputs.where((s) => s.isNotEmpty);
    for (final child in attrs.values.whereType<OnnxSession>()) {
      yield* child._captures;
    }
  }

  factory _Node.read(_Proto p, int opset, Map<String, Uint8List> externalData,
      _FunctionScope functions) {
    final domain = p.string(7);
    final op = domain.isEmpty ? p.string(4) : '$domain::${p.string(4)}';
    final name = p.string(3);
    final label = name.isEmpty ? op : '$name ($op)';
    final call = functions.resolve(p, opset, externalData);
    if (call != null)
      return _Node(op, label, p.strings(2), p.strings(1), {}, call);
    final history = _schemas[op];
    final schema = _extensions[op] ??
        history?.entries.where((e) => e.key <= opset).lastOrNull?.value;
    if (schema == null)
      throw UnsupportedError('$label: unsupported operator at opset $opset');
    final outs = p.strings(2), ins = p.strings(1);
    schema.check(ins, outs, label);
    final attrs = <String, Object>{};
    for (final entry in p.messages(5)) {
      final key = entry.string(1);
      final a = functions.bind(entry);
      final kind = schema.attributes[key];
      if (kind == null) {
        throw UnsupportedError(
            '$label: unsupported attribute $key at opset $opset');
      }
      if (attrs.containsKey(key) || a.integer(20) != kind)
        throw FormatException('$label: invalid attribute $key');
      attrs[key] = switch (kind) {
        1 => ByteData.sublistView(a.bytes(2)).getFloat32(0, Endian.little),
        2 => a.integer(3),
        3 => a.string(4),
        4 => _tensor(a.message(5), externalData: externalData),
        5 => OnnxSession._readGraph(a.message(6), opset,
            allowCapture: true,
            externalData: externalData,
            functions: functions),
        11 => _sparseTensor(a.message(22), externalData),
        13 => TensorInfo._type(key, a.message(14)),
        6 => [
            for (final field in a.all(7))
              for (var i = 0; i < (field as Uint8List).length; i += 4)
                ByteData.sublistView(field).getFloat32(i, Endian.little)
          ],
        7 => a.ints(8).toList(),
        8 => a.strings(9),
        _ => throw UnsupportedError('$label: attribute $key'),
      };
    }
    for (final required in schema.requiredAttributes) {
      if (!attrs.containsKey(required))
        throw FormatException('$label: missing attribute $required');
    }
    return _Node(op, label, outs, ins, attrs);
  }
  int integer(String name, int fallback) => attrs[name] as int? ?? fallback;
  double real(String name, double fallback) =>
      attrs[name] as double? ?? fallback;
  List<int>? ints(String name) => attrs[name] as List<int>?;
}

class _Schema {
  final int minInputs, maxInputs, minOutputs, maxOutputs;
  final String inputs, outputs;
  final Map<String, int> attributes;
  final List<String> requiredAttributes;
  const _Schema(this.minInputs, this.maxInputs, this.inputs, this.minOutputs,
      this.maxOutputs, this.outputs, this.attributes, this.requiredAttributes);

  void check(List<String> ins, List<String> outs, String label) {
    void checkSlots(
        List<String> names, String slots, int min, int max, String role) {
      if (names.length < min || names.length > max)
        throw FormatException('$label: invalid number of $role');
      for (var i = 0; i < names.length; i++) {
        if (names[i].isEmpty && (i >= slots.length || slots[i] != '?'))
          throw FormatException('$label: missing $role at $i');
      }
    }

    checkSlots(ins, inputs, minInputs, maxInputs, 'inputs');
    checkSlots(outs, outputs, minOutputs, maxOutputs, 'outputs');
  }
}

const _matMulNBitsSchema = _Schema(
    3,
    4,
    '111?',
    1,
    1,
    '1',
    {'K': 2, 'N': 2, 'bits': 2, 'block_size': 2, 'accuracy_level': 2},
    ['K', 'N', 'bits', 'block_size']);

const _extensions = <String, _Schema>{
  'com.microsoft::MatMulNBits': _matMulNBitsSchema,
  'com.microsoft::GatherBlockQuantized': _Schema(3, 4, '111?', 1, 1, '1',
      {'bits': 2, 'block_size': 2, 'gather_axis': 2, 'quantize_axis': 2}, []),
  'SimplifiedLayerNormalization': _Schema(
      2, 2, '11', 1, 2, '1?', {'axis': 2, 'epsilon': 1, 'stash_type': 2}, []),
  'com.microsoft::SkipSimplifiedLayerNormalization':
      _Schema(3, 4, '111?', 1, 4, '1???', {'epsilon': 1}, []),
  'com.microsoft::RotaryEmbedding': _Schema(4, 4, '1111', 1, 1, '1',
      {'num_heads': 2, 'rotary_embedding_dim': 2, 'interleaved': 2}, []),
  'com.microsoft::MultiHeadAttention': _Schema(1, 8, '1???????', 1, 3, '1??',
      {'num_heads': 2, 'scale': 1}, ['num_heads']),
  'com.microsoft::GroupQueryAttention':
      _Schema(7, 11, '111??11????', 1, 3, '1??', {
    'num_heads': 2,
    'kv_num_heads': 2,
    'scale': 1,
    'softcap': 1,
    'local_window_size': 2,
    'do_rotary': 2,
    'rotary_interleaved': 2
  }, [
    'num_heads',
    'kv_num_heads'
  ]),
};

// An embedding table can also be the output projection. Read its contiguous
// rows directly to avoid a second, vocabulary-sized buffer for Transpose.
const _rowProduct = '#MatMulTransposedRhs';
List<_Node> _fuseTransposedWeights(
    List<_Node> nodes,
    Map<String, Tensor> weights,
    Map<String, TensorInfo> inputs,
    Map<String, TensorInfo> outputs,
    Map<String, int> uses) {
  final candidates = <String, _Node>{};
  final replacements = <_Node, _Node>{}, removed = <_Node>{};
  for (final n in nodes) {
    if (n.op == 'Transpose') {
      final name = n.inputs.single, w = weights[name], out = n.outputs.single;
      final perm = n.ints('perm');
      if (w?.type == TensorType.float32 &&
          w!.shape.length == 2 &&
          !inputs.containsKey(name) &&
          !outputs.containsKey(out) &&
          uses[out] == 1 &&
          (perm == null || _sameShape(perm, [1, 0]))) candidates[out] = n;
    } else if (n.op == 'MatMul') {
      final transpose = candidates[n.inputs[1]];
      if (transpose == null) continue;
      replacements[n] = _Node(_rowProduct, n.label, n.outputs,
          [n.inputs[0], transpose.inputs.single], {});
      removed.add(transpose);
      uses.remove(n.inputs[1]);
    }
  }
  return [
    for (final n in nodes)
      if (!removed.contains(n)) replacements[n] ?? n
  ];
}

// Exporters express exact GELU as five standard nodes. Fuse only private
// intermediates and scalar FLOAT initializers that callers cannot override.
const _expandedGelu = '#ExpandedGelu';
List<_Node> _fuseGelu(
    List<_Node> nodes,
    Map<String, Tensor> weights,
    Map<String, TensorInfo> inputs,
    Map<String, TensorInfo> outputs,
    Map<String, int> uses) {
  final private = <String, _Node>{
    for (final n in nodes)
      if (n.function == null &&
          n.outputs.length == 1 &&
          uses[n.outputs.single] == 1 &&
          !outputs.containsKey(n.outputs.single))
        n.outputs.single: n,
  };
  final constants = <String, Tensor>{
    for (final n in nodes)
      if (n.op == 'Constant' &&
          n.function == null &&
          n.attrs['value'] is Tensor)
        n.outputs.single: n.attrs['value'] as Tensor,
  };
  bool scalar(String name, double value) {
    final w = weights[name] ?? constants[name];
    return w != null &&
        w.type == TensorType.float32 &&
        w.shape.isEmpty &&
        !inputs.containsKey(name) &&
        w.values.single == value;
  }

  final removed = <_Node>{}, replacements = <_Node, _Node>{};
  for (final end in nodes) {
    if (end.op != 'Mul' || end.function != null || !scalar(end.inputs[1], 0.5))
      continue;
    final product = private[end.inputs[0]];
    if (product == null || product.op != 'Mul') continue;
    final x = product.inputs.first, plus = private[product.inputs[1]];
    if (plus == null || plus.op != 'Add' || !scalar(plus.inputs[1], 1))
      continue;
    final erf = private[plus.inputs.first];
    if (erf == null || erf.op != 'Erf') continue;
    final div = private[erf.inputs.single];
    if (div == null ||
        div.op != 'Div' ||
        div.inputs.first != x ||
        !scalar(div.inputs[1], _geluDivisor)) continue;
    removed.addAll([div, erf, plus, product]);
    replacements[end] = _Node(_expandedGelu, end.label, end.outputs, [x], {});
  }
  if (replacements.isEmpty) return nodes;
  final plan = [
    for (final n in nodes)
      if (!removed.contains(n)) replacements[n] ?? n
  ];
  uses.clear();
  for (final n in plan) {
    for (final key in n.references) {
      uses.update(key, (count) => count + 1, ifAbsent: () => 1);
    }
  }
  return plan;
}
