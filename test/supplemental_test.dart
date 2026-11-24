import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_onnx/flutter_onnx.dart';
import 'package:test/test.dart';
import 'supplemental_models.dart';

OnnxSession open(String name) =>
    OnnxSession.fromBytes(base64Decode(supplementalModels[name]!));

void main() {
  test('integer Gemm combines both fractions before truncating', () {
    final large = (BigInt.one << 58) + BigInt.one;
    final y = open('integer_gemm').run({
      'a': Tensor.bigInts(TensorType.int64, [2, 1], [large, -large]),
      'b': Tensor.int64([1, 2], [2, 1]),
      'c': Tensor.int64([2], [0, -2]),
    })['y']!;
    expect(y.bigIntValues, [
      large,
      (large + BigInt.one) ~/ BigInt.two,
      -large,
      (-large + BigInt.one) ~/ BigInt.two
    ]);
  });
  test('unrecognized image bytes return an empty image', () {
    final y = open('image_decode').run({
      'encoded': Tensor.fromValues(TensorType.uint8, [3], [0, 17, 99]),
    })['pixels']!;
    expect(y.shape, [0, 0, 3]);
    expect(y.values, isEmpty);
  });
  test('complex tensors preserve real and imaginary components', () {
    for (final type in [TensorType.complex64, TensorType.complex128]) {
      final values = [(1.25, -8.5), (-0.5, 4.0)];
      final tensor = Tensor.complex(type, [2], values);
      expect(
          Tensor.fromBytes(type, [2], tensor.toBytes()).complexValues, values);
    }
  });
  test('a model-local function binds a referenced attribute', () {
    final out = open('function').run({
      'x': Tensor.float32([2], [-4, 2])
    });
    expect(out['y']!.values, [-.5, 2]);
  });
  test('sparse COO constant fills unspecified coordinates with zero', () {
    final out = open('sparse').run({
      'x': Tensor.float32([2, 2], [1, 2, 3, 4])
    });
    expect(out['y']!.values, [1, 7, 0, 4]);
  });
  test('float64 precision and exact wide INT64 matrix products and addition',
      () {
    final doubleOut = open('double').run({
      'x': Tensor.fromValues(TensorType.float64, [1, 2], [1 + 1e-11, 1]),
      'w': Tensor.fromValues(TensorType.float64, [2, 1], [1, -1]),
      'bias': Tensor.fromValues(TensorType.float64, [], [2e-11])
    });
    expect(doubleOut['out']!.values.single, closeTo(3e-11, 1e-16));
    final large = (BigInt.one << 56) + BigInt.one;
    final wideOut = open('integer').run({
      'x': Tensor.bigInts(TensorType.int64, [1, 2], [large, BigInt.one]),
      'w': Tensor.int64([2, 1], [3, -2]),
      'bias': Tensor.int64([], [7])
    });
    expect(wideOut['out']!.bigIntValues,
        [large * BigInt.from(3) + BigInt.from(5)]);
  });
  test(
      'sequence construction, insertion, erasure, selection, concat and optional round trip',
      () {
    final y = open('containers').run({
      'a': Tensor.float32([2], [1, 3]),
      'b': Tensor.float32([2], [5, 7]),
      'idx': Tensor.int64([], [1])
    });
    expect(y['length']!.values, [1]);
    expect(y['at']!.values, [1, 3]);
    expect(y['joined']!.values, [1, 3, 1, 3]);
    expect(y['present']!.values, [5, 7]);
  });
  test('ROI maxima and global L2 norm', () {
    final data = List.generate(16, (i) => i - 8);
    final y = open('pool').run({
      'x': Tensor.float32([1, 1, 4, 4], data),
      'rois': Tensor.float32([1, 5], [0, 0, 0, 3, 3])
    });
    expect(y['roi']!.values, [-3, -1, 5, 7]);
    expect(y['norm']!.values.single, closeTo(math.sqrt(344), 1e-5));
  });
  test(
      'seeded random operators reproduce draws and satisfy distribution and mask properties',
      () {
    const count = 12000;
    final feeds = {
      'x': Tensor.float32([count], List.filled(count, 0.25)),
      'logits': Tensor.float32([1, 2], [0, math.log(3)]),
      'ratio': Tensor.float32([], [0.5]),
      'training': Tensor.fromValues(TensorType.boolean, [], [1])
    };
    final session = open('random');
    final first = session.run(feeds),
        next = session.run(feeds),
        replay = open('random').run(feeds);
    for (final e in first.entries) {
      final values = e.value.values;
      expect(values, replay[e.key]!.values, reason: e.key);
      expect(values, isNot(equals(next[e.key]!.values)), reason: e.key);
      expect(e.value.shape, e.key == 'classes' ? [1, count] : [count]);
      final mean = values.fold<double>(0, (a, b) => a + b) / count;
      if (e.key.startsWith('RandomNormal')) {
        expect(mean, closeTo(0, .05));
        expect(values.fold<double>(0, (sum, v) => sum + v * v) / count,
            closeTo(1, .07));
      } else if (e.key.startsWith('RandomUniform')) {
        expect(values.every((v) => v >= 0 && v < 1), isTrue);
        expect(mean, closeTo(.5, .02));
      } else if (e.key != 'drop') {
        expect(values.every((v) => v == 0 || v == 1), isTrue);
        expect(mean,
            closeTo({'mask': .5, 'coin': .25, 'classes': .75}[e.key]!, .03));
      }
    }
    expect(first['classes']!.type, TensorType.int32);
    expect(first['mask']!.type, TensorType.boolean);
    expect(first['drop']!.values, first['mask']!.values.map((v) => v * .5));
  });
  test(
      'external initializer uses its byte range and detects missing or truncated files',
      () {
    final model = base64Decode(supplementalModels['external']!);
    final sidecar = ByteData(14)
      ..setFloat32(3, 1.25, Endian.little)
      ..setFloat32(7, -2.5, Endian.little);
    final session = OnnxSession.fromBytes(model,
        externalData: {'weights.bin': sidecar.buffer.asUint8List()});
    expect(
        session.run({
          'x': Tensor.float32([2], [2, 3])
        })['y']!.values,
        [3.25, .5]);
    expect(() => OnnxSession.fromBytes(model), throwsFormatException);
    expect(
        () => OnnxSession.fromBytes(model,
            externalData: {'weights.bin': Uint8List(10)}),
        throwsFormatException);
  });
  test(
      'INT64 and UINT64 byte encodings survive web and native without precision loss',
      () {
    for (final t in [TensorType.int64, TensorType.uint64]) {
      final values = t == TensorType.uint64
          ? [BigInt.zero, (BigInt.one << 64) - BigInt.one]
          : [-(BigInt.one << 63), (BigInt.one << 63) - BigInt.one];
      final original = Tensor.bigInts(t, [2], values);
      expect(Tensor.fromBytes(t, [2], original.toBytes()).bigIntValues, values);
    }
  });
}
