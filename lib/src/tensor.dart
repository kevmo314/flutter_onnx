part of '../flutter_onnx.dart';

enum TensorType {
  float32(1, 4),
  uint8(2, 1),
  int8(3, 1),
  uint16(4, 2),
  int16(5, 2),
  int32(6, 4),
  int64(7, 8),
  boolean(9, 1),
  float16(10, 2),
  float64(11, 8),
  uint32(12, 4),
  uint64(13, 8),
  string(8, 0),
  complex64(14, 8),
  complex128(15, 16),
  bfloat16(16, 2),
  float8e4m3fn(17, 1),
  float8e4m3fnuz(18, 1),
  float8e5m2(19, 1),
  float8e5m2fnuz(20, 1),
  uint4(21, 0),
  int4(22, 0),
  float4e2m1(23, 0),
  float8e8m0(24, 1),
  uint2(25, 0),
  int2(26, 0),
  float6e2m3(27, 0),
  float6e3m2(28, 0);

  int get bitWidth => switch (this) {
        uint2 || int2 => 2,
        uint4 || int4 || float4e2m1 => 4,
        float6e2m3 || float6e3m2 => 6,
        _ => bytesPerElement * 8
      };
  bool get complex => this == complex64 || this == complex128;
  bool get compact => onnxId >= 17;
  bool get smallInteger =>
      this == int4 || this == uint4 || this == int2 || this == uint2;

  const TensorType(this.onnxId, this.bytesPerElement);
  final int onnxId, bytesPerElement;
  bool get floating =>
      this == float32 ||
      this == float16 ||
      this == float64 ||
      this == bfloat16 ||
      (compact && !smallInteger);
}

/// A dense, immutable tensor stored in ONNX row-major order.
class Tensor implements OnnxValue {
  final TensorType type;
  final List<int> shape;
  final List<num>? _numbers;
  final List<String>? _strings;
  final List<BigInt>? _wide;
  final List<(double, double)>? _pairs;

  /// ONNX complex elements in (real, imaginary) order.
  List<(double, double)> get complexValues =>
      _pairs ?? (throw StateError('Expected complex tensor'));
  Tensor._complex(this.type, List<int> dims, this._pairs)
      : shape = List.unmodifiable(dims),
        _wide = null,
        _numbers = null,
        _strings = null {
    _require(_size(shape) == length, 'Complex data length differs from shape');
  }
  factory Tensor.complex(
      TensorType type, List<int> shape, Iterable<(double, double)> values) {
    _require(type.complex, 'Expected COMPLEX64 or COMPLEX128');
    final tmp = Float32List(2);
    return Tensor._complex(type, shape, List.unmodifiable(values.map((v) {
      if (type == TensorType.complex128) return v;
      tmp[0] = v.$1;
      tmp[1] = v.$2;
      return (tmp[0], tmp[1]);
    })));
  }

  /// Exact signed or unsigned 64-bit elements on both Dart VM and JavaScript.
  List<BigInt> get bigIntValues {
    if (type != TensorType.int64 && type != TensorType.uint64)
      throw StateError('Expected a 64-bit integer tensor');
    return _wide ??
        List<BigInt>.unmodifiable(values.map((n) => BigInt.from(n)));
  }

  Tensor._wideInts(this.type, List<int> dims, this._wide)
      : shape = List.unmodifiable(dims),
        _pairs = null,
        _strings = null,
        _numbers = null {
    _require(_size(shape) == length, 'Integer data length differs from shape');
  }
  factory Tensor.bigInts(
      TensorType type, List<int> shape, Iterable<BigInt> values) {
    _require(type == TensorType.int64 || type == TensorType.uint64,
        'Expected INT64 or UINT64');
    final data = List<BigInt>.unmodifiable(values);
    final signed = type == TensorType.int64;
    _require(
        data.every((v) => (signed ? v.toSigned(64) : v.toUnsigned(64)) == v),
        'Integer exceeds tensor element range');
    if (data.every((v) => v.abs() <= BigInt.from(_maxSafe)))
      return Tensor.fromValues(type, shape, data.map((v) => v.toInt()));
    return Tensor._wideInts(type, shape, data);
  }
  List<num> get values =>
      _numbers ??
      (throw StateError(type.complex
          ? 'Use complexValues for a complex tensor'
          : _wide == null
              ? 'Use stringValues for a STRING tensor'
              : 'Use bigIntValues for a wide integer tensor'));
  List<String> get stringValues =>
      _strings ?? (throw StateError('Expected STRING tensor'));
  int get length =>
      _numbers?.length ?? _pairs?.length ?? _wide?.length ?? _strings!.length;
  Tensor.strings(List<int> shape, Iterable<String> values)
      : type = TensorType.string,
        _pairs = null,
        shape = List.unmodifiable(shape),
        _strings = List.unmodifiable(values),
        _numbers = null,
        _wide = null {
    _require(_size(shape) == length, 'String data length differs from shape');
  }

  Tensor._(this.type, List<int> shape, this._numbers)
      : _strings = null,
        _pairs = null,
        _wide = null,
        shape = List.unmodifiable(shape) {
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

  /// Construct a tensor of the given numeric type. BOOL uses zero and one.
  factory Tensor.fromValues(
      TensorType type, List<int> shape, Iterable<num> values) {
    if (type == TensorType.string) throw ArgumentError('Use Tensor.strings');
    if (type.compact) return _compactFromValues(type, shape, values);
    if (type == TensorType.float32) return Tensor.float32(shape, values);
    if (type == TensorType.int64)
      return Tensor.int64(shape, values.map((v) => v.toInt()));
    if (type == TensorType.float64)
      return Tensor._(
          type,
          shape,
          Float64List.fromList(values.map((v) => v.toDouble()).toList())
              .asUnmodifiableView());
    if (type == TensorType.float16 || type == TensorType.bfloat16) {
      final store = ByteData(4);
      return Tensor._(
          type,
          shape,
          Float32List.fromList(values
                  .map((v) => _decodeHalf(
                      _encodeHalf(v.toDouble(), type, store), type, store))
                  .toList())
              .asUnmodifiableView());
    }
    if (type == TensorType.boolean)
      return Tensor._(
          type,
          shape,
          Uint8List.fromList(values.map((v) => v == 0 ? 0 : 1).toList())
              .asUnmodifiableView());
    final ints = values.map((v) => v.toInt()).toList();
    final List<int> data = switch (type) {
      TensorType.uint8 => Uint8List.fromList(ints).asUnmodifiableView(),
      TensorType.int8 => Int8List.fromList(ints).asUnmodifiableView(),
      TensorType.uint16 => Uint16List.fromList(ints).asUnmodifiableView(),
      TensorType.int16 => Int16List.fromList(ints).asUnmodifiableView(),
      TensorType.int32 => Int32List.fromList(ints).asUnmodifiableView(),
      TensorType.uint32 => Uint32List.fromList(ints).asUnmodifiableView(),
      TensorType.uint64 => ints.every((v) => v >= 0 && v <= _maxSafe)
          ? List<int>.unmodifiable(ints)
          : throw UnsupportedError('UINT64 exceeds portable range'),
      TensorType.boolean =>
        Uint8List.fromList(ints.map((v) => v == 0 ? 0 : 1).toList())
            .asUnmodifiableView(),
      _ => throw StateError(type.name)
    };
    return Tensor._(type, shape, data);
  }

  /// Decode the little-endian raw_data encoding used by ONNX.
  factory Tensor.fromBytes(TensorType type, List<int> shape, Uint8List bytes) =>
      _decodeRaw(type, shape, bytes);

  /// Returns an independent copy of the little-endian ONNX raw_data bytes.
  Uint8List toBytes() {
    if (type == TensorType.string)
      throw UnsupportedError('STRING uses ONNX string_data');
    if (type.compact) return _encodeCompact(this);
    if (type.complex) {
      final out = ByteData(length * type.bytesPerElement);
      final narrow = type == TensorType.complex64;
      for (var i = 0; i < length; i++) {
        final p = i * type.bytesPerElement, value = complexValues[i];
        if (narrow) {
          out.setFloat32(p, value.$1, Endian.little);
          out.setFloat32(p + 4, value.$2, Endian.little);
        } else {
          out.setFloat64(p, value.$1, Endian.little);
          out.setFloat64(p + 8, value.$2, Endian.little);
        }
      }
      return out.buffer.asUint8List();
    }
    if (_wide != null) {
      final out = ByteData(length * 8);
      for (var i = 0; i < length; i++) {
        final b = _wide[i].toUnsigned(64);
        out.setUint32(i * 8, b.toUnsigned(32).toInt(), Endian.little);
        out.setUint32(i * 8 + 4, (b >> 32).toInt(), Endian.little);
      }
      return out.buffer.asUint8List();
    }

    if (values is TypedData &&
        type != TensorType.float16 &&
        type != TensorType.bfloat16 &&
        Endian.host == Endian.little) {
      final src = values as TypedData;
      return Uint8List.fromList(
          src.buffer.asUint8List(src.offsetInBytes, src.lengthInBytes));
    }
    final out = ByteData(values.length * type.bytesPerElement),
        store = ByteData(4);
    for (var i = 0; i < values.length; i++) {
      final p = i * type.bytesPerElement, v = values[i];
      switch (type) {
        case TensorType.float16:
        case TensorType.bfloat16:
          out.setUint16(
              p, _encodeHalf(v.toDouble(), type, store), Endian.little);
        case TensorType.int64:
        case TensorType.uint64:
          final b = BigInt.from(v.toInt()).toUnsigned(64);
          out.setUint32(p, b.toUnsigned(32).toInt(), Endian.little);
          out.setUint32(p + 4, (b >> 32).toInt(), Endian.little);
        case TensorType.float32:
          out.setFloat32(p, v.toDouble(), Endian.little);
        case TensorType.float64:
          out.setFloat64(p, v.toDouble(), Endian.little);
        case TensorType.uint8:
        case TensorType.int8:
        case TensorType.boolean:
          out.setUint8(p, v.toInt());
        case TensorType.uint16:
        case TensorType.int16:
          out.setUint16(p, v.toInt(), Endian.little);
        case TensorType.uint32:
        case TensorType.int32:
          out.setUint32(p, v.toInt(), Endian.little);
        default:
          throw StateError(type.name);
      }
    }
    return out.buffer.asUint8List();
  }

  Tensor _reshape(List<int> dims) => type.complex
      ? Tensor._complex(type, dims, _pairs)
      : type == TensorType.string
          ? Tensor.strings(dims, stringValues)
          : _wide != null
              ? Tensor._wideInts(type, dims, _wide)
              : Tensor._(type, dims, values);
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
  if (x.type.complex)
    return Tensor._complex(
        x.type,
        shape,
        List.unmodifiable(
            List.generate(_size(shape), (i) => x.complexValues[offset(i)])));
  if (x._wide != null)
    return Tensor.bigInts(
        x.type, shape, List.generate(_size(shape), (i) => x._wide[offset(i)]));
  if (x.type == TensorType.string)
    return Tensor.strings(
        shape, List.generate(_size(shape), (i) => x.stringValues[offset(i)]));
  if (x.type == TensorType.float32) {
    return _float(shape, (i) => x.values[offset(i)].toDouble());
  }
  return Tensor.fromValues(
      x.type, shape, List.generate(_size(shape), (i) => x.values[offset(i)]));
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

Tensor _decodeRaw(TensorType type, List<int> shape, Uint8List bytes,
    {bool sliceBound = false}) {
  final count = _size(shape),
      width = type.bytesPerElement,
      d = ByteData.sublistView(bytes),
      store = ByteData(4);
  if (type.compact) {
    if (bytes.length != (count * type.bitWidth + 7) ~/ 8)
      throw const FormatException('Wrong packed tensor length');
    return Tensor._(
        type,
        shape,
        List<num>.unmodifiable(List.generate(
            count,
            (i) => _compactValue(
                type, _readPacked(bytes, i * type.bitWidth, type.bitWidth)))));
  }
  if (bytes.length != count * width)
    throw const FormatException('Incorrect raw_data length');
  if (type.complex) {
    final narrow = type == TensorType.complex64;
    return Tensor._complex(
        type,
        shape,
        List.unmodifiable(List.generate(count, (i) {
          final p = i * width;
          return narrow
              ? (
                  d.getFloat32(p, Endian.little),
                  d.getFloat32(p + 4, Endian.little)
                )
              : (
                  d.getFloat64(p, Endian.little),
                  d.getFloat64(p + 8, Endian.little)
                );
        })));
  }
  if (type == TensorType.float32)
    return _float(shape, (i) => d.getFloat32(i * 4, Endian.little));
  if (type == TensorType.uint8)
    return Tensor._(
        type, shape, Uint8List.fromList(bytes).asUnmodifiableView());
  if (type == TensorType.float16 || type == TensorType.bfloat16) {
    final buf = Float32List(count);
    for (var i = 0; i < count; i++) {
      buf[i] = _decodeHalf(d.getUint16(i * 2, Endian.little), type, store);
    }
    return Tensor._(type, shape, buf.asUnmodifiableView());
  }
  if (type == TensorType.int64 || type == TensorType.uint64) {
    final numbers = List.generate(count, (i) {
      final p = i * 8;
      final hi = type == TensorType.int64
          ? d.getInt32(p + 4, Endian.little)
          : d.getUint32(p + 4, Endian.little);
      final value =
          (BigInt.from(hi) << 32) | BigInt.from(d.getUint32(p, Endian.little));
      if (sliceBound && value.abs() > BigInt.from(_maxSafe))
        return BigInt.from(value.sign * _maxSafe);
      return value;
    });
    return Tensor.bigInts(type, shape, numbers);
  }
  num read(int i) {
    final p = i * width;
    switch (type) {
      case TensorType.float64:
        return d.getFloat64(p, Endian.little);
      case TensorType.int32:
        return d.getInt32(p, Endian.little);
      case TensorType.uint32:
        return d.getUint32(p, Endian.little);
      case TensorType.int16:
        return d.getInt16(p, Endian.little);
      case TensorType.uint16:
        return d.getUint16(p, Endian.little);
      case TensorType.int8:
        return d.getInt8(p);
      case TensorType.boolean:
        return d.getUint8(p);
      default:
        throw StateError(type.name);
    }
  }

  return Tensor.fromValues(type, shape, List.generate(count, read));
}

double _decodeHalf(int bits, TensorType type, ByteData scratch) {
  if (type == TensorType.bfloat16) {
    scratch.setUint32(0, bits * 65536);
    return scratch.getFloat32(0);
  }
  final sign = (bits & 32768) == 0 ? 1.0 : -1.0,
      e = (bits >> 10) & 31,
      f = bits & 1023;
  if (e == 31) return f == 0 ? sign * double.infinity : double.nan;
  if (e == 0) return sign * f / 16777216;
  return sign * (1 + f / 1024) * math.pow(2, e - 15);
}

int _encodeHalf(double value, TensorType type, ByteData scratch) {
  scratch.setFloat32(0, value);
  final bits = scratch.getUint32(0), high = bits ~/ 65536;
  if (type == TensorType.bfloat16) {
    if (value.isNaN) return high | 64;
    return (bits + 32767 + (high & 1)) ~/ 65536 & 65535;
  }
  final sign = high & 32768, exp = (high >> 7) & 255, f = bits & 0x7fffff;
  if (exp == 255) return sign | (f == 0 ? 0x7c00 : 0x7e00);
  if (exp > 142) return sign | 0x7c00;
  if (exp < 102) return sign;
  final shift = exp < 113 ? 126 - exp : 13;
  final significand = exp < 113 ? 0x800000 | f : f, unit = 1 << shift;
  var rounded = significand ~/ unit;
  final remainder = significand % unit;
  if (remainder * 2 > unit || (remainder * 2 == unit && rounded.isOdd))
    rounded++;
  return sign | ((exp < 113 ? 0 : (exp - 112) * 1024) + rounded);
}
