part of '../flutter_onnx.dart';

/// Tensor interface declared by an ONNX graph. Dimensions may be integers,
/// symbolic names or null (unknown). A null dimensions list means unknown rank.
class TensorInfo {
  final String name;
  final TensorType type;
  final List<Object?>? dimensions;
  final String kind;
  final TensorInfo? element;
  TensorInfo._(this.name, this.type, this.dimensions,
      [this.kind = "tensor", this.element]);

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
    return OnnxSession._(
        version,
        inputs,
        outputs,
        weights,
        _fuseTransposedWeights(nodes, weights, inputs, outputs, uses),
        uses,
        captures);
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
    for (final node in _nodes) {
      final args = frame.args(node);
      final product =
          (node.op == 'MatMul' || node.op == 'Gemm' || node.op == _rowProduct)
              ? await _tryAccelerate(node, args.cast<Tensor?>(), backend)
              : null;
      if (product != null) onAccelerated?.call(node.label);
      frame.store(node,
          product == null ? _valueNode(node, args, frame.values) : [product]);
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

Future<Tensor?> _tryAccelerate(
    _Node node, List<Tensor?> args, MatMulBackend backend) async {
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
