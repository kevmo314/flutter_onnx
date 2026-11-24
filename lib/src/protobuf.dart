part of '../flutter_onnx.dart';

// A length-delimited field is a view into the model, including for raw weights.
class _Reader {
  final Uint8List bytes;
  int offset = 0;
  _Reader(this.bytes);

  Uint8List take(int size) {
    if (size < 0 || size > bytes.length - offset) {
      throw const FormatException('Truncated protobuf field');
    }
    final start = offset;
    offset += size;
    return Uint8List.sublistView(bytes, start, offset);
  }

  int integer() {
    // Small tags and lengths dominate. Wide signed values need BigInt on web.
    var result = 0;
    var factor = 1;
    for (var i = 0; i < 7; i++) {
      final b = _byte();
      result += (b % 128) * factor;
      if (b < 128) return result;
      factor *= 128;
    }
    var wide = BigInt.from(result);
    for (var i = 7; i < 10; i++) {
      final b = _byte();
      if (i == 9 && b > 1) throw const FormatException('Invalid int64');
      wide |= BigInt.from(b % 128) << (7 * i);
      if (b < 128) return _portableInt(wide.toSigned(64));
    }
    throw const FormatException('Invalid varint');
  }

  int _byte() {
    if (offset >= bytes.length) throw const FormatException('Truncated varint');
    return bytes[offset++];
  }
}

int _portableInt(BigInt n) {
  if (n.abs() > BigInt.from(_maxSafe)) {
    throw UnsupportedError('INT64 value exceeds the portable integer range');
  }
  return n.toInt();
}

class _Proto {
  final fields = <int, List<Object>>{};
  _Proto(Uint8List bytes) {
    final r = _Reader(bytes);
    while (r.offset < bytes.length) {
      final tag = r.integer();
      if (tag < 8 || tag > 0xffffffff)
        throw const FormatException('Invalid field tag');
      final value = switch (tag % 8) {
        0 => r.integer(),
        1 => r.take(8),
        2 => r.take(r.integer()),
        5 => r.take(4),
        _ => throw const FormatException('Unsupported protobuf wire type'),
      };
      (fields[tag ~/ 8] ??= []).add(value);
    }
  }
  List<Object> all(int id) => fields[id] ?? const [];
  bool has(int id) => fields.containsKey(id);
  Object one(int id) {
    final values = all(id);
    if (values.length != 1)
      throw FormatException('Expected one protobuf field $id');
    return values.single;
  }

  Uint8List bytes(int id) => one(id) as Uint8List;
  int integer(int id) => one(id) as int;
  String string(int id, [String fallback = '']) =>
      has(id) ? utf8.decode(bytes(id)) : fallback;
  _Proto message(int id) => _Proto(bytes(id));
  Iterable<_Proto> messages(int id) =>
      all(id).map((v) => _Proto(v as Uint8List));
  List<String> strings(int id) =>
      all(id).map((v) => utf8.decode(v as Uint8List)).toList();
  Iterable<int> ints(int id) sync* {
    for (final field in all(id)) {
      if (field is int) {
        yield field;
      } else {
        final r = _Reader(field as Uint8List);
        while (r.offset < r.bytes.length) {
          yield r.integer();
        }
      }
    }
  }
}

Tensor _tensor(_Proto p, {bool sliceBound = false}) {
  if (p.has(3) || p.has(13) || (p.has(14) && p.integer(14) != 0)) {
    throw UnsupportedError('External or segmented tensor data');
  }
  final shape = p.ints(1).toList();
  final type = _type(p.integer(2));
  if (type == TensorType.string) return Tensor.strings(shape, p.strings(6));
  if (p.has(9))
    return _decodeRaw(type, shape, p.bytes(9), sliceBound: sliceBound);
  if (type.compact) {
    final codes = p.ints(5).toList();
    if (type.bitWidth == 2 || type.bitWidth == 4)
      return _decodeRaw(type, shape, Uint8List.fromList(codes));
    return Tensor._(type, shape,
        List<num>.unmodifiable(codes.map((v) => _compactValue(type, v))));
  }
  if (type == TensorType.int64 ||
      type == TensorType.uint64 ||
      type == TensorType.uint32) {
    return Tensor.fromValues(
        type, shape, p.ints(type == TensorType.int64 ? 7 : 11));
  }
  if (type == TensorType.float16 || type == TensorType.bfloat16) {
    final store = ByteData(4);
    return Tensor.fromValues(
        type, shape, p.ints(5).map((v) => _decodeHalf(v, type, store)));
  }
  if (!type.floating) return Tensor.fromValues(type, shape, p.ints(5));
  final values = <double>[];
  for (final chunk in p.all(type == TensorType.float32 ? 4 : 10)) {
    final data = ByteData.sublistView(chunk as Uint8List),
        width = type.bytesPerElement;
    if (data.lengthInBytes % width != 0)
      throw const FormatException('Misaligned typed tensor values');
    for (var i = 0; i < data.lengthInBytes; i += width) {
      values.add(width == 4
          ? data.getFloat32(i, Endian.little)
          : data.getFloat64(i, Endian.little));
    }
  }
  return Tensor.fromValues(type, shape, values);
}

TensorType _type(int id) => TensorType.values.firstWhere((t) => t.onnxId == id,
    orElse: () => throw UnsupportedError('ONNX tensor element type $id'));
