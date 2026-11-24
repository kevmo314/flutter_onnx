part of '../flutter_onnx.dart';

/// Tensor interface declared by an ONNX graph. Dimensions may be integers,
/// symbolic names or null (unknown). A null dimensions list means unknown rank.
class TensorInfo {
  final String name;
  final TensorType type;
  final List<Object?>? dimensions;
  TensorInfo._(this.name, this.type, this.dimensions);

  factory TensorInfo._read(_Proto value) {
    final tensor = value.message(2);
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
    return TensorInfo._(value.string(1), _type(t.integer(1)), dims);
  }

  void _check(Tensor t, Map<String, int> symbols) {
    _require(type == t.type, '$name expects $type, got ${t.type}');
    final dims = dimensions;
    if (dims == null) return;
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
  OnnxSession._(this.opset, this.inputs, this.outputs, this._weights,
      this._nodes, this._uses);

  factory OnnxSession.fromBytes(Uint8List bytes) {
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
      if (model.has(25)) throw UnsupportedError('Model-local functions');
      final graph = model.message(7);
      if (graph.has(15)) throw UnsupportedError('Sparse initializers');
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
      final nodes =
          graph.messages(1).map((p) => _Node.read(p, version)).toList();
      final sliceBounds = <String>{};
      final otherUses = {...inputs.keys, ...outputs.keys};
      for (final n in nodes) {
        for (final (i, name) in n.inputs.indexed) {
          if (n.op == 'Slice' && (i == 1 || i == 2)) {
            sliceBounds.add(name);
          } else {
            otherUses.add(name);
          }
        }
      }
      sliceBounds.removeAll(otherUses);
      final weights = <String, Tensor>{};
      for (final t in graph.messages(5)) {
        final name = t.string(8);
        if (name.isEmpty || weights.containsKey(name))
          throw FormatException('Invalid initializer name "$name"');
        weights[name] = _tensor(t, sliceBound: sliceBounds.contains(name));
      }
      final known = {...inputs.keys, ...weights.keys};
      final uses = <String, int>{};
      for (final node in nodes) {
        for (final name in node.inputs.where((n) => n.isNotEmpty)) {
          if (!known.contains(name))
            throw FormatException('${node.label}: unknown input $name');
          uses.update(name, (v) => v + 1, ifAbsent: () => 1);
        }
        for (final name in node.outputs.where((v) => v.isNotEmpty)) {
          if (!known.add(name))
            throw FormatException('Duplicate graph value $name');
        }
      }
      for (final name in outputs.keys) {
        if (!known.contains(name))
          throw FormatException('Unknown graph output $name');
      }
      return OnnxSession._(version, inputs, outputs, weights, nodes, uses);
    } on TypeError catch (e) {
      throw FormatException('Invalid ONNX protobuf field: $e');
    }
  }

  Map<String, Tensor> run(Map<String, Tensor> feeds) {
    final frame = _Frame(this, feeds);
    for (final node in _nodes) {
      frame.store(node, _executeNode(node, frame.args(node)));
    }
    return frame.results();
  }

  /// Offloads supported matrix products and executes other nodes locally.
  /// [onAccelerated] is called only after a backend produces a result.
  Future<Map<String, Tensor>> runAsync(Map<String, Tensor> feeds,
      {MatMulBackend? backend,
      void Function(String node)? onAccelerated}) async {
    if (backend == null) return run(feeds);
    final frame = _Frame(this, feeds);
    for (final node in _nodes) {
      final args = frame.args(node);
      final product = await _tryAccelerate(node, args, backend);
      if (product != null) onAccelerated?.call(node.label);
      frame.store(node, product == null ? _executeNode(node, args) : [product]);
    }
    return frame.results();
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
  final Map<String, Tensor> values;
  final Map<String, int> remaining;
  _Frame(this.session, Map<String, Tensor> feeds)
      : values = {...session._weights},
        remaining = {...session._uses} {
    for (final name in feeds.keys) {
      _require(session.inputs.containsKey(name), 'Unknown graph input $name');
    }
    for (final info in session.inputs.values) {
      final value = feeds[info.name] ?? session._weights[info.name];
      if (value == null)
        throw ArgumentError('Missing graph input ${info.name}');
      info._check(value, symbols);
      values[info.name] = value;
    }
  }

  List<Tensor?> args(_Node n) => n.inputs.map((key) => values[key]).toList();
  void store(_Node n, List<Tensor> outputs) {
    for (final (i, key) in n.outputs.indexed) {
      if (key.isNotEmpty &&
          (remaining.containsKey(key) || session.outputs.containsKey(key)))
        values[key] = outputs[i];
    }
    for (final k in n.inputs.where((s) => s.isNotEmpty)) {
      if ((remaining[k] = remaining[k]! - 1) == 0 &&
          !session.outputs.containsKey(k)) values.remove(k);
    }
  }

  Map<String, Tensor> results() {
    final result = <String, Tensor>{};
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
  _Node(this.op, this.label, this.outputs, this.inputs, this.attrs);

  factory _Node.read(_Proto p, int opset) {
    final domain = p.string(7);
    final op = domain.isEmpty ? p.string(4) : '$domain::${p.string(4)}';
    final name = p.string(3);
    final label = name.isEmpty ? op : '$name ($op)';
    final schema = _schemas[op];
    if (schema == null)
      throw UnsupportedError('$label: unsupported operator domain or type');
    final outs = p.strings(2);
    if (outs.isEmpty ||
        outs.first.isEmpty ||
        (!const {
              'Attention',
              'RMSNormalization',
              'LayerNormalization',
              'Split',
              'TopK',
              'Dropout',
              'CausalConvWithState',
              'LinearAttention',
              'DynamicQuantizeLinear'
            }.contains(op) &&
            outs.skip(1).any((s) => s.isNotEmpty)))
      throw UnsupportedError('$label: requires one output');
    final ins = p.strings(1);
    if (ins.length < schema.$1 ||
        ins.length > schema.$2 ||
        ins.take(schema.$1).contains('')) {
      throw FormatException('$label: invalid input list');
    }
    final attrs = <String, Object>{};
    for (final a in p.messages(5)) {
      final key = a.string(1);
      final kind = schema.$3[key];
      if (kind == null ||
          (opset < 14 && (key == 'allowzero' || key == 'training_mode'))) {
        throw UnsupportedError(
            '$label: unsupported attribute $key at opset $opset');
      }
      if (attrs.containsKey(key) || a.integer(20) != kind)
        throw FormatException('$label: invalid attribute $key');
      attrs[key] = switch (kind) {
        1 => ByteData.sublistView(a.bytes(2)).getFloat32(0, Endian.little),
        2 => a.integer(3),
        3 => a.string(4),
        4 => _tensor(a.message(5)),
        7 => a.ints(8).toList(),
        _ => throw UnsupportedError('$label: attribute $key'),
      };
    }
    return _Node(op, label, outs, ins, attrs);
  }
  int integer(String name, int fallback) => attrs[name] as int? ?? fallback;
  double real(String name, double fallback) =>
      attrs[name] as double? ?? fallback;
  List<int>? ints(String name) => attrs[name] as List<int>?;
}

// (required input count, maximum input count, AttributeProto types).
const _schemas = <String, (int, int, Map<String, int>)>{
  'Attention': (
    3,
    7,
    {
      'q_num_heads': 2,
      'kv_num_heads': 2,
      'is_causal': 2,
      'scale': 1,
      'softcap': 1,
      'softmax_precision': 2,
      'qk_matmul_output_mode': 2,
      'left_window_size': 2,
      'right_window_size': 2
    }
  ),
  'RMSNormalization': (2, 2, {'axis': 2, 'epsilon': 1, 'stash_type': 2}),
  'LayerNormalization': (2, 3, {'axis': 2, 'epsilon': 1, 'stash_type': 2}),
  'RotaryEmbedding': (
    3,
    4,
    {'num_heads': 2, 'rotary_embedding_dim': 2, 'interleaved': 2}
  ),
  'Gelu': (1, 1, {'approximate': 3}),
  'Erf': (1, 1, {}),
  'SwiGLU': (2, 2, {'alpha': 1}),
  'Split': (1, 2, {'axis': 2, 'num_outputs': 2}),
  'ArgMax': (1, 1, {'axis': 2, 'keepdims': 2, 'select_last_index': 2}),
  'ArgMin': (1, 1, {'axis': 2, 'keepdims': 2, 'select_last_index': 2}),
  'TopK': (2, 2, {'axis': 2, 'largest': 2, 'sorted': 2}),
  'Dropout': (1, 3, {'seed': 2}),
  'com.microsoft::MatMulNBits': (
    3,
    4,
    {'K': 2, 'N': 2, 'bits': 2, 'block_size': 2, 'accuracy_level': 2}
  ),
  'CastLike': (2, 2, {'saturate': 2, 'round_mode': 3}),
  'BitCast': (1, 1, {'to': 2}),
  'BitwiseNot': (1, 1, {}),
  'BitShift': (2, 2, {'direction': 3}),
  'BitwiseAnd': (2, 2, {}),
  'BitwiseOr': (2, 2, {}),
  'BitwiseXor': (2, 2, {}),
  'Mod': (2, 2, {'fmod': 2}),
  'IsInf': (1, 1, {'detect_negative': 2, 'detect_positive': 2}),
  'CumSum': (2, 2, {'exclusive': 2, 'reverse': 2}),
  'CumProd': (2, 2, {'exclusive': 2, 'reverse': 2}),
  'EyeLike': (1, 1, {'dtype': 2, 'k': 2}),
  'GatherND': (2, 2, {'batch_dims': 2}),
  'Compress': (2, 2, {'axis': 2}),
  'NonZero': (1, 1, {}),
  'OneHot': (3, 3, {'axis': 2}),
  'ReverseSequence': (2, 2, {'time_axis': 2, 'batch_axis': 2}),
  'DepthToSpace': (1, 1, {'blocksize': 2, 'mode': 3}),
  'SpaceToDepth': (1, 1, {'blocksize': 2}),
  'Pad': (2, 4, {'mode': 3}),
  'CenterCropPad': (2, 2, {'axes': 7}),
  'Scatter': (3, 3, {'axis': 2, 'reduction': 3}),
  'ScatterElements': (3, 3, {'axis': 2, 'reduction': 3}),
  'HannWindow': (1, 1, {'periodic': 2, 'output_datatype': 2}),
  'HammingWindow': (1, 1, {'periodic': 2, 'output_datatype': 2}),
  'BlackmanWindow': (1, 1, {'periodic': 2, 'output_datatype': 2}),
  'Shrink': (1, 1, {'bias': 1, 'lambd': 1}),
  'Hardmax': (1, 1, {'axis': 2}),
  'GlobalMaxPool': (1, 1, {}),
  'GlobalLpPool': (1, 1, {'axis': 2, 'p': 2}),
  'LpNormalization': (1, 1, {'axis': 2, 'p': 2}),
  'InstanceNormalization': (3, 3, {'epsilon': 1}),
  'GroupNormalization': (
    3,
    3,
    {'num_groups': 2, 'epsilon': 1, 'stash_type': 2}
  ),
  'MeanVarianceNormalization': (1, 1, {'axes': 7}),
  'LRN': (1, 1, {'alpha': 1, 'beta': 1, 'bias': 1, 'size': 2}),
  'Det': (1, 1, {}),
  'MatMulInteger': (2, 4, {}),
  'QLinearMatMul': (8, 8, {}),
  'QuantizeLinear': (
    2,
    3,
    {
      'axis': 2,
      'saturate': 2,
      'block_size': 2,
      'output_dtype': 2,
      'precision': 2
    }
  ),
  'DequantizeLinear': (
    2,
    3,
    {
      'axis': 2,
      'saturate': 2,
      'block_size': 2,
      'output_dtype': 2,
      'precision': 2
    }
  ),
  'DynamicQuantizeLinear': (1, 1, {}),
  'Clip': (1, 3, {}),
  'HardSigmoid': (1, 1, {'alpha': 1, 'beta': 1}),
  'HardSwish': (1, 1, {}),
  'Reciprocal': (1, 1, {}),
  'Floor': (1, 1, {}),
  'Ceil': (1, 1, {}),
  'Round': (1, 1, {}),
  'Cos': (1, 1, {}),
  'Sin': (1, 1, {}),
  'Tan': (1, 1, {}),
  'Acos': (1, 1, {}),
  'Asin': (1, 1, {}),
  'Atan': (1, 1, {}),
  'Cosh': (1, 1, {}),
  'Sinh': (1, 1, {}),
  'Acosh': (1, 1, {}),
  'Asinh': (1, 1, {}),
  'Atanh': (1, 1, {}),
  'Sign': (1, 1, {}),
  'Softsign': (1, 1, {}),
  'Softplus': (1, 1, {}),
  'IsNaN': (1, 1, {}),
  'Not': (1, 1, {}),
  'Elu': (1, 1, {'alpha': 1}),
  'Celu': (1, 1, {'alpha': 1}),
  'ThresholdedRelu': (1, 1, {'alpha': 1}),
  'Swish': (1, 1, {'alpha': 1}),
  'Selu': (1, 1, {'alpha': 1, 'gamma': 1}),
  'Pow': (2, 2, {}),
  'Equal': (2, 2, {}),
  'Greater': (2, 2, {}),
  'GreaterOrEqual': (2, 2, {}),
  'Less': (2, 2, {}),
  'LessOrEqual': (2, 2, {}),
  'And': (2, 2, {}),
  'Or': (2, 2, {}),
  'Xor': (2, 2, {}),
  'PRelu': (2, 2, {}),
  'Sum': (1, 2147483647, {}),
  'Mean': (1, 2147483647, {}),
  'Min': (1, 2147483647, {}),
  'Max': (1, 2147483647, {}),
  'Shape': (1, 1, {'start': 2, 'end': 2}),
  'Size': (1, 1, {}),
  'Gather': (2, 2, {'axis': 2}),
  'GatherElements': (2, 2, {'axis': 2}),
  'Expand': (2, 2, {}),
  'Tile': (2, 2, {}),
  'Where': (3, 3, {}),
  'Range': (3, 3, {}),
  'Squeeze': (1, 2, {}),
  'Unsqueeze': (2, 2, {}),
  'Cast': (1, 1, {'to': 2, 'saturate': 2}),
  'ConstantOfShape': (1, 1, {'value': 4}),
  'ScatterND': (3, 3, {'reduction': 3}),
  'Slice': (3, 5, {}),
  'Trilu': (1, 2, {'upper': 2}),
  'ReduceMean': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceSum': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceMax': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceMin': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceProd': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceL1': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceL2': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceLogSum': (1, 2, {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}),
  'ReduceLogSumExp': (
    1,
    2,
    {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}
  ),
  'ReduceSumSquare': (
    1,
    2,
    {'axes': 7, 'keepdims': 2, 'noop_with_empty_axes': 2}
  ),
  'LogSoftmax': (1, 1, {'axis': 2}),
  'Identity': (1, 1, {}),
  'Relu': (1, 1, {}),
  'Sigmoid': (1, 1, {}),
  'Tanh': (1, 1, {}),
  'Exp': (1, 1, {}),
  'Sqrt': (1, 1, {}),
  'Abs': (1, 1, {}),
  'Neg': (1, 1, {}),
  'Log': (1, 1, {}),
  'LeakyRelu': (1, 1, {'alpha': 1}),
  'Add': (2, 2, {}),
  'Sub': (2, 2, {}),
  'Mul': (2, 2, {}),
  'Div': (2, 2, {}),
  'MatMul': (2, 2, {}),
  'Gemm': (2, 3, {'alpha': 1, 'beta': 1, 'transA': 2, 'transB': 2}),
  'Softmax': (1, 1, {'axis': 2}),
  'Flatten': (1, 1, {'axis': 2}),
  'Reshape': (2, 2, {'allowzero': 2}),
  'Transpose': (1, 1, {'perm': 7}),
  'Constant': (0, 0, {'value': 4}),
  'Concat': (1, 0x7fffffff, {'axis': 2}),
  'Conv': (
    2,
    3,
    {
      'kernel_shape': 7,
      'auto_pad': 3,
      'pads': 7,
      'strides': 7,
      'dilations': 7,
      'group': 2
    }
  ),
  'MaxPool': (
    1,
    1,
    {
      'kernel_shape': 7,
      'auto_pad': 3,
      'pads': 7,
      'strides': 7,
      'dilations': 7,
      'ceil_mode': 2,
      'storage_order': 2
    }
  ),
  'AveragePool': (
    1,
    1,
    {
      'kernel_shape': 7,
      'auto_pad': 3,
      'pads': 7,
      'strides': 7,
      'ceil_mode': 2,
      'count_include_pad': 2
    }
  ),
  'GlobalAveragePool': (1, 1, {}),
  'BatchNormalization': (
    5,
    5,
    {'epsilon': 1, 'momentum': 1, 'training_mode': 2}
  ),
};
