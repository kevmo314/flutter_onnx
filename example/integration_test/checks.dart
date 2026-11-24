import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnx_accelerators/flutter_onnx_accelerators.dart';

Future<OnnxSession> _load(String file) async {
  final data = await rootBundle.load('assets/$file.onnx');
  return OnnxSession.fromBytes(
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
}

Future<Map<String, Object?>> verifyInference() async {
  final attention = await _load('attention');
  final fixture =
      jsonDecode(await rootBundle.loadString('assets/attention.json')) as Map;
  final outputs = attention.run({
    for (final e in (fixture['feeds'] as Map).entries)
      e.key: Tensor.float32((e.value['shape'] as List).cast<int>(),
          (e.value['values'] as List).cast<num>())
  });
  for (final e in (fixture['outputs'] as Map).entries) {
    final actual = outputs[e.key]!;
    final expected = (e.value['values'] as List).cast<num>();
    if (!listEquals(actual.shape, (e.value['shape'] as List).cast<int>()) ||
        expected.length != actual.length)
      throw StateError('Shape mismatch for ${e.key}');
    for (var i = 0; i < expected.length; i++) {
      if (!((actual.values[i] - expected[i]).abs() <= 1e-5)) {
        throw StateError('${e.key}[$i]: ${actual.values[i]} vs ${expected[i]}');
      }
    }
  }
  final session = await _load('xor');
  final backend = await createMatMulBackend();
  var dispatched = false;
  try {
    if (backend != null) {
      final result = await backend.multiply(
          Float32List.fromList([1, -2, 3, 0, 4, 2]),
          Float32List.fromList([2, 1, 0, 3, 4, -1]),
          rows: 2,
          inner: 3,
          columns: 2);
      if (result != null || backend.name == 'Metal Performance Shaders') {
        if (!listEquals<num>(result, [14, -8, 8, 10]))
          throw StateError('Incorrect ${backend.name} matrix result: $result');
        dispatched = true;
      }
    }
    for (final a in [0, 1]) {
      for (final b in [0, 1]) {
        final out = await session.runAsync({
          'bits': Tensor.float32([1, 2], [a, b])
        }, backend: backend);
        if (!((out['xor']!.values.single - (a ^ b)).abs() <= 1e-5))
          throw StateError('Incorrect XOR result for $a,$b');
      }
    }
    return {
      'backend': backend?.name,
      'matrixDispatched': dispatched,
      'attentionOutputs': outputs.length,
      'xorCases': 4
    };
  } finally {
    await backend?.close();
  }
}
