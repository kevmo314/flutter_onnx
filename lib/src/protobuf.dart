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
    final value = varint();
    return value is int ? value : _portableInt(value as BigInt);
  }

  Object varint() {
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
      if (b < 128) return wide.toSigned(64);
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
        0 => r.varint(),
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
  int integer(int id) {
    final value = one(id);
    return value is int ? value : _portableInt(value as BigInt);
  }

  String string(int id, [String fallback = '']) =>
      has(id) ? utf8.decode(bytes(id)) : fallback;
  _Proto message(int id) => _Proto(bytes(id));
  Iterable<_Proto> messages(int id) =>
      all(id).map((v) => _Proto(v as Uint8List));
  List<String> strings(int id) =>
      all(id).map((v) => utf8.decode(v as Uint8List)).toList();
  Iterable<BigInt> wideInts(int id) sync* {
    BigInt wide(Object v) => v is BigInt ? v : BigInt.from(v as int);
    for (final field in all(id)) {
      if (field is Uint8List) {
        final reader = _Reader(field);
        while (reader.offset < field.length) yield wide(reader.varint());
      } else {
        yield wide(field);
      }
    }
  }

  Iterable<int> ints(int id) sync* {
    for (final field in all(id)) {
      if (field is int || field is BigInt) {
        yield field is int ? field : _portableInt(field as BigInt);
      } else {
        final r = _Reader(field as Uint8List);
        while (r.offset < r.bytes.length) {
          yield r.integer();
        }
      }
    }
  }
}

Tensor _tensor(_Proto p,
    {bool sliceBound = false, Map<String, Uint8List> externalData = const {}}) {
  if (p.has(3)) throw UnsupportedError('Segmented tensor data');
  final shape = p.ints(1).toList();
  final type = _type(p.integer(2));
  if (p.has(13) || (p.has(14) && p.integer(14) != 0)) {
    if (!p.has(14) || p.integer(14) != 1 || p.has(9)) {
      throw const FormatException('Invalid external tensor storage');
    }
    final entries = <String, String>{};
    for (final e in p.messages(13)) {
      final key = e.string(1);
      if (entries.containsKey(key))
        throw FormatException('Duplicate external tensor key $key');
      entries[key] = e.string(2);
    }
    final location = entries['location'];
    final file = externalData[location];
    if (file == null)
      throw FormatException('Missing external tensor file $location');
    int field(String key, int fallback) {
      final text = entries[key];
      if (text == null) return fallback;
      final value = int.tryParse(text);
      if (value == null || value < 0)
        throw FormatException('Invalid external tensor $key');
      return value;
    }

    final start = field('offset', 0),
        length = field('length', file.length - start);
    if (start > file.length || length < 0 || length > file.length - start)
      throw FormatException('External tensor range exceeds $location');
    return _decodeRaw(
        type, shape, Uint8List.sublistView(file, start, start + length),
        sliceBound: sliceBound);
  }
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
  if (type == TensorType.int64 || type == TensorType.uint64) {
    final signed = type == TensorType.int64;
    final integers = p.wideInts(signed ? 7 : 11).map((v) {
      final value = signed ? v.toSigned(64) : v.toUnsigned(64);
      return sliceBound && value.abs() > BigInt.from(_maxSafe)
          ? BigInt.from(value.sign * _maxSafe)
          : value;
    });
    return Tensor.bigInts(type, shape, integers);
  }
  if (type == TensorType.uint32)
    return Tensor.fromValues(type, shape, p.ints(11));
  if (type == TensorType.float16 || type == TensorType.bfloat16) {
    final store = ByteData(4);
    return Tensor.fromValues(
        type, shape, p.ints(5).map((v) => _decodeHalf(v, type, store)));
  }
  if (!type.floating && !type.complex)
    return Tensor.fromValues(type, shape, p.ints(5));
  final values = <double>[];
  final single = type == TensorType.float32 || type == TensorType.complex64;
  for (final chunk in p.all(single ? 4 : 10)) {
    final data = ByteData.sublistView(chunk as Uint8List),
        width = single ? 4 : 8;
    if (data.lengthInBytes % width != 0)
      throw const FormatException('Misaligned typed tensor values');
    for (var i = 0; i < data.lengthInBytes; i += width) {
      values.add(width == 4
          ? data.getFloat32(i, Endian.little)
          : data.getFloat64(i, Endian.little));
    }
  }
  if (type.complex) {
    if (values.length != _size(shape) * 2)
      throw const FormatException('Wrong number of complex components');
    return Tensor.complex(type, shape, [
      for (var i = 0; i < values.length; i += 2) (values[i], values[i + 1])
    ]);
  }
  return Tensor.fromValues(type, shape, values);
}

TensorType _type(int id) => TensorType.values.firstWhere((t) => t.onnxId == id,
    orElse: () => throw UnsupportedError('ONNX tensor element type $id'));

Tensor _sparseTensor(_Proto sparse, Map<String, Uint8List> externalData) {
  final values = _tensor(sparse.message(1), externalData: externalData);
  final indices = _tensor(sparse.message(2), externalData: externalData);
  final shape = sparse.ints(3).toList(),
      length = _size(shape),
      count = values.length;
  _require(values.shape.length == 1 && indices.type == TensorType.int64,
      'Invalid sparse tensor components');
  final flat = indices.shape.length == 1;
  _require(_sameShape(indices.shape, flat ? [count] : [count, shape.length]),
      'Invalid sparse index shape');
  final strides = _strides(shape), positions = <int, int>{};
  var prior = -1;
  for (var i = 0; i < count; i++) {
    var offset = 0;
    if (flat) {
      offset = indices.values[i].toInt();
    } else {
      for (var j = 0; j < shape.length; j++) {
        final k = indices.values[i * shape.length + j].toInt();
        _require(k >= 0 && k < shape[j], 'Sparse index outside dimension');
        offset += k * strides[j];
      }
    }
    _require(offset > prior && offset < length,
        'Sparse indices must be ordered and in bounds');
    prior = offset;
    positions[offset] = i;
  }
  List<T> expand<T>(List<T> data, T zero) => List.generate(length, (i) {
        final pos = positions[i];
        return pos == null ? zero : data[pos];
      });
  if (values.type == TensorType.string)
    return Tensor.strings(shape, expand(values.stringValues, ''));
  if (values.type.complex)
    return Tensor.complex(
        values.type, shape, expand(values.complexValues, (0.0, 0.0)));
  if (values._wide != null)
    return Tensor.bigInts(
        values.type, shape, expand(values.bigIntValues, BigInt.zero));
  return Tensor.fromValues(values.type, shape, expand(values.values, 0));
}
