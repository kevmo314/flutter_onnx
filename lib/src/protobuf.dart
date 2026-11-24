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

Tensor _tensor(_Proto p) {
  if (p.has(3) || p.has(13) || (p.has(14) && p.integer(14) != 0)) {
    throw UnsupportedError('External or segmented tensor data');
  }
  final shape = p.ints(1).toList();
  final count = _size(shape);
  final type = _type(p.integer(2));
  if (p.has(9)) {
    final data = ByteData.sublistView(p.bytes(9));
    final width = type == TensorType.float32 ? 4 : 8;
    if (data.lengthInBytes != count * width)
      throw const FormatException('Incorrect raw_data length');
    if (type == TensorType.float32)
      return _float(shape, (i) => data.getFloat32(i * 4, Endian.little));
    return Tensor.int64(
        shape,
        List.generate(count, (i) {
          final lo = data.getUint32(i * 8, Endian.little);
          final hi = data.getInt32(i * 8 + 4, Endian.little);
          return _portableInt((BigInt.from(hi) << 32) + BigInt.from(lo));
        }));
  }
  if (type == TensorType.int64) return Tensor.int64(shape, p.ints(7));
  final numbers = <double>[];
  for (final chunk in p.all(4)) {
    final data = ByteData.sublistView(chunk as Uint8List);
    if (data.lengthInBytes % 4 != 0)
      throw const FormatException('Invalid float_data length');
    for (var i = 0; i < data.lengthInBytes; i += 4) {
      numbers.add(data.getFloat32(i, Endian.little));
    }
  }
  return Tensor.float32(shape, numbers);
}

TensorType _type(int id) => switch (id) {
      1 => TensorType.float32,
      7 => TensorType.int64,
      _ => throw UnsupportedError('ONNX tensor element type $id'),
    };
