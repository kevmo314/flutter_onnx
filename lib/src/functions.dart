part of '../flutter_onnx.dart';

typedef _FunctionId = (String, String, String);

class _FunctionScope {
  final Map<_FunctionId, _Proto> definitions;
  final Set<_FunctionId> active;
  final Map<String, _Proto> attributes;
  const _FunctionScope(
      [this.definitions = const {},
      this.active = const {},
      this.attributes = const {}]);
  factory _FunctionScope.fromModel(_Proto model) {
    final definitions = <_FunctionId, _Proto>{};
    for (final f in model.messages(25)) {
      final id = (f.string(10), f.string(1), f.string(13));
      if (definitions.containsKey(id))
        throw FormatException('Duplicate ONNX function $id');
      definitions[id] = f;
    }
    return _FunctionScope(definitions);
  }
  _Proto bind(_Proto a) {
    if (!a.has(21)) return a;
    final key = a.string(21), value = attributes[key];
    if (value == null || value.integer(20) != a.integer(20))
      throw FormatException('Missing or mistyped function attribute $key');
    return value;
  }

  _FunctionCall? resolve(
      _Proto node, int opset, Map<String, Uint8List> externalData) {
    final id = (node.string(7), node.string(4), node.string(8)),
        f = definitions[id];
    if (f == null) return null;
    if (active.contains(id))
      throw FormatException('Recursive ONNX function $id');
    final bound = {for (final a in f.messages(11)) a.string(1): a};
    final required = f.strings(6), allowed = {...required, ...bound.keys};
    for (final a in node.messages(5)) {
      final key = a.string(1);
      if (!allowed.contains(key))
        throw FormatException('$id has no attribute $key');
      bound[key] = bind(a);
    }
    if (required.any((k) => !bound.containsKey(k)))
      throw FormatException('Missing required attribute in $id');
    final inner = _FunctionScope(definitions, {...active, id}, bound);
    final standard =
        f.messages(9).where((i) => i.string(1).isEmpty).firstOrNull;
    final version = standard?.integer(2) ?? opset;
    if (version < 13 || version > 28)
      throw UnsupportedError('Function $id imports opset $version');
    final inputs = f.strings(4), outputs = f.strings(5);
    if (inputs.length != node.strings(1).length ||
        outputs.length != node.strings(2).length)
      throw FormatException('Function $id argument or output count mismatch');
    return _FunctionCall(inputs, outputs, [
      for (final n in f.messages(7)) _Node.read(n, version, externalData, inner)
    ]);
  }
}

class _FunctionCall {
  final List<String> inputs, outputs;
  final List<_Node> nodes;
  _FunctionCall(this.inputs, this.outputs, this.nodes);
  List<OnnxValue> run(List<OnnxValue?> args) {
    final local = <String, OnnxValue>{
      for (final (i, k) in inputs.indexed)
        if (args[i] != null) k: args[i]!
    };
    for (final n in nodes) {
      final out = _valueNode(n, [for (final k in n.inputs) local[k]], local);
      for (final (i, k) in n.outputs.indexed) {
        if (k.isNotEmpty) local[k] = out[i];
      }
    }
    return [for (final k in outputs) local[k]!];
  }
}
