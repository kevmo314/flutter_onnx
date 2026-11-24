part of '../flutter_onnx.dart';

enum TensorType { float32, int64 }

/// A dense, immutable tensor stored in ONNX row-major order.
class Tensor {
  final TensorType type;
  final List<int> shape;
  final List<num> values;

  Tensor._(this.type, List<int> shape, this.values)
      : shape = List.unmodifiable(shape) {
    if (_size(shape) != values.length) {
      throw ArgumentError(
          'Shape $shape does not match ${values.length} values');
    }
  }

  factory Tensor.float32(List<int> shape, Iterable<num> values) => Tensor._(
      TensorType.float32,
      shape,
      Float32List.fromList(values.map((v) => v.toDouble()).toList())
          .asUnmodifiableView());

  /// Values must fit the exact integer range shared by Dart VM and JavaScript.
  factory Tensor.int64(List<int> shape, Iterable<int> values) {
    final data = List<int>.unmodifiable(values);
    if (data.any((n) => n < -_maxSafe || n > _maxSafe)) {
      throw ArgumentError('INT64 value exceeds the portable integer range');
    }
    return Tensor._(TensorType.int64, shape, data);
  }
  Tensor _reshape(List<int> dims) => Tensor._(type, dims, values);
}

const _maxSafe = 9007199254740991;

int _size(Iterable<int> dims) {
  if (dims.any((d) => d < 0 || d > 0x7fffffff)) {
    throw ArgumentError('Invalid tensor dimensions $dims');
  }
  if (dims.contains(0)) return 0;
  var count = 1;
  for (final dim in dims) {
    if (count > 0x7fffffff ~/ dim) throw ArgumentError('Tensor too large');
    count *= dim;
  }
  return count;
}

Tensor _float(List<int> shape, double Function(int) value) {
  final data = Float32List(_size(shape));
  for (var i = 0; i < data.length; i++) {
    data[i] = value(i);
  }
  return Tensor._(TensorType.float32, shape, data.asUnmodifiableView());
}

Tensor _reorder(Tensor x, List<int> shape, int Function(int) offset) {
  if (x.type == TensorType.float32) {
    return _float(shape, (i) => x.values[offset(i)].toDouble());
  }
  return Tensor.int64(
      shape, List.generate(_size(shape), (i) => x.values[offset(i)].toInt()));
}

List<int> _strides(List<int> dims) {
  final strides = List.filled(dims.length, 1);
  for (var i = dims.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

int _axis(int value, int rank, {bool allowEnd = false}) {
  final ax = value < 0 ? rank + value : value;
  if (ax < 0 || ax >= rank + (allowEnd ? 1 : 0)) {
    throw ArgumentError('Axis $value outside rank $rank');
  }
  return ax;
}

void _require(bool condition, String message) {
  if (!condition) throw ArgumentError(message);
}
