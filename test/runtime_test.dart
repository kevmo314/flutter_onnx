import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:test/test.dart';
import 'vectors.dart';
import 'supplemental_models.dart';

Tensor tensor(List<dynamic> spec) {
  final dims = (spec[1] as List).cast<int>();
  final data = (spec[2] as List).cast<num>();
  return spec[0] == 'int64'
      ? Tensor.int64(dims, data.cast<int>())
      : Tensor.float32(dims, data);
}

class _Backend implements MatMulBackend {
  final bool decline;
  _Backend({this.decline = false});
  @override
  String get name => 'test matrix kernel';
  @override
  Future<void> close() async {}
  @override
  Future<Float32List?> multiply(Float32List left, Float32List right,
      {required int rows, required int inner, required int columns}) async {
    if (decline) return null;
    final out = Float32List(rows * columns);
    for (var i = 0; i < rows; i++) {
      for (var j = 0; j < columns; j++) {
        for (var k = 0; k < inner; k++) {
          out[i * columns + j] += left[i * inner + k] * right[k * columns + j];
        }
      }
    }
    return out;
  }
}

void main() {
  test(
      'expanded GELU preserves rounding, exposed intermediates and overridable constants',
      () {
    final fused = OnnxSession.fromBytes(base64Decode(
        'CA46sgEKEAoBeAoBZBIDZGl2IgNEaXYKDwoDZGl2EgNlcmYiA0VyZgoUCgNlcmYKA29uZRIDYWRkIgNBZGQKEgoBeAoDYWRkEgNtdWwiA011bAoTCgNtdWwKBGhhbGYSAXkiA011bBIEZ2VsdSoLEAEiBPMEtT9CAWQqDRABIgQAAIA/QgNvbmUqDhABIgQAAAA/QgRoYWxmWg0KAXgSCAoGCAESAgoAYg0KAXkSCAoGCAESAgoAQgQKABAS'));
    final exposed = OnnxSession.fromBytes(base64Decode(
        'CA460wEKEAoBeAoBZBIDZGl2IgNEaXYKDwoDZGl2EgNlcmYiA0VyZgoUCgNlcmYKA29uZRIDYWRkIgNBZGQKEgoBeAoDYWRkEgNtdWwiA011bAoTCgNtdWwKBGhhbGYSAXkiA011bBIEZ2VsdSoLEAEiBPMEtT9CAWQqDRABIgQAAIA/QgNvbmUqDhABIgQAAAA/QgRoYWxmWg0KAXgSCAoGCAESAgoAWg4KBGhhbGYSBgoECAESAGINCgF5EggKBggBEgIKAGIPCgNlcmYSCAoGCAESAgoAQgQKABAS'));
    final x = Tensor.float32([513], List.generate(513, (i) => (i - 256) / 37));
    final actual = fused.run({'x': x})['y']!;
    final reference = exposed.run({'x': x});
    expect(actual.toBytes(), reference['y']!.toBytes());
    expect(reference['erf']!.values[256], 0);
    final scaled = exposed.run({
      'x': x,
      'half': Tensor.float32([], [2])
    });
    expect(scaled['y']!.values, actual.values.map((v) => v * 4).toList());
  });

  test('tied weights produce the same projection on CPU and device', () async {
    final session = OnnxSession.fromBytes(
        base64Decode(supplementalModels['tied_weights']!));
    final feeds = {
      'x': Tensor.float32([2, 2], [1, 3, 4, 0])
    };
    final expected = [-2, 14, 3, 4, 8, -12], dispatched = <String>[];
    expect(session.run(feeds)['scores']!.values, expected);
    final result = await session.runAsync(feeds,
        backend: _Backend(), onAccelerated: dispatched.add);
    expect(result['scores']!.values, expected);
    expect(dispatched, ['MatMul']);
  });
  final cases = jsonDecode(vectorsJson) as List;
  for (final c in cases) {
    test('ONNX reference: ${c['name']}', () async {
      final s = OnnxSession.fromBytes(base64Decode(c['model']));
      final feeds =
          (c['feeds'] as Map).map((k, v) => MapEntry(k as String, tensor(v)));
      final observed = <String>[];
      for (final actual in [
        s.run(feeds),
        await s.runAsync(feeds,
            backend: _Backend(), onAccelerated: observed.add)
      ]) {
        for (final e in (c['outputs'] as Map).entries) {
          final expected = tensor(e.value), got = actual[e.key]!;
          expect(got.shape, expected.shape);
          expect(got.type, expected.type);
          expect(got.values.length, expected.values.length);
          for (var i = 0; i < got.values.length; i++) {
            expect(
                got.values[i],
                closeTo(expected.values[i],
                    1e-5 + 1e-5 * expected.values[i].abs()));
          }
        }
      }
      if (c['name'] == 'xor') expect(observed, ['Gemm', 'MatMul']);
    });
  }
  final bytes = base64Decode(cases.first['model']);
  final session = OnnxSession.fromBytes(bytes);
  test('declined device kernel falls back for every call', () async {
    for (final a in [0, 1]) {
      for (final b in [0, 1]) {
        final r = await session.runAsync({
          'bits': Tensor.float32([1, 2], [a, b])
        },
            backend: _Backend(decline: true),
            onAccelerated: (_) => fail('unexpected device result'));
        expect(r['xor']!.values.single, a ^ b);
      }
    }
  });
  test('validates graph inputs', () {
    expect(() => session.run({}), throwsArgumentError);
    expect(
        () => session.run({
              'bits': Tensor.float32([2], [1, 2])
            }),
        throwsArgumentError);
    expect(
        () => session.run({
              'bits': Tensor.int64([1, 2], [1, 2])
            }),
        throwsArgumentError);
    expect(
        () => session.run({
              'extra': Tensor.float32([], [1])
            }),
        throwsArgumentError);
  });
  test('rejects truncation and unknown operators', () {
    expect(() => OnnxSession.fromBytes(bytes.sublist(0, bytes.length - 1)),
        throwsFormatException);
    final model = latin1.decode(bytes).replaceFirst('Gemm', 'NoOp');
    expect(
        () => OnnxSession.fromBytes(Uint8List.fromList(latin1.encode(model))),
        throwsUnsupportedError);
  });
  test('tensor data and shape are immutable', () {
    final dims = [2], data = [1, 2];
    final t = Tensor.float32(dims, data);
    dims[0] = 1;
    data[0] = 0;
    expect(t.shape, [2]);
    expect(t.values, [1, 2]);
    expect(() => t.values[0] = 0.0, throwsUnsupportedError);
    expect(() => t.shape[0] = 0, throwsUnsupportedError);
  });
}
