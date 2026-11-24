part of '../flutter_onnx.dart';

int _readPacked(Uint8List bytes, int bit, int width) {
  final pos = bit ~/ 8, offset = bit % 8;
  final word = bytes[pos] + (offset + width > 8 ? bytes[pos + 1] * 256 : 0);
  return (word >> offset) & ((1 << width) - 1);
}

num _compactValue(TensorType t, int code) {
  final bits = t.bitWidth, mask = (1 << bits) - 1;
  code &= mask;
  if (t.smallInteger)
    return t.name.startsWith('int') ? code.toSigned(bits) : code;
  if (t == TensorType.float8e8m0)
    return code == 255 ? double.nan : math.pow(2.0, code - 127);
  final eBits = switch (t) {
    TensorType.float8e5m2 || TensorType.float8e5m2fnuz => 5,
    TensorType.float8e4m3fn || TensorType.float8e4m3fnuz => 4,
    TensorType.float6e3m2 => 3,
    _ => 2
  };
  final mBits = bits - eBits - 1,
      mMask = (1 << mBits) - 1,
      eMask = (1 << eBits) - 1;
  final uz = t == TensorType.float8e4m3fnuz || t == TensorType.float8e5m2fnuz;
  final sign = code >> (bits - 1) == 0 ? 1.0 : -1.0,
      bias = (1 << (eBits - 1)) - (uz ? 0 : 1);
  final exp = (code >> mBits) & eMask, frac = code & mMask;
  if (uz && code == (1 << (bits - 1))) return double.nan;
  if (t == TensorType.float8e4m3fn && exp == eMask && frac == mMask)
    return double.nan;
  if (t == TensorType.float8e5m2 && exp == eMask)
    return frac == 0 ? sign * double.infinity : double.nan;
  if (exp == 0) return sign * frac / (1 << mBits) * math.pow(2, 1 - bias);
  return sign * (1 + frac / (1 << mBits)) * math.pow(2, exp - bias);
}

final _tables = <TensorType, List<num>>{};
int _compactCode(TensorType t, num value,
    {bool saturate = true, String rounding = 'up'}) {
  if (t.smallInteger) return value.toInt().toUnsigned(t.bitWidth);
  final entries = _tables.putIfAbsent(
      t, () => List.generate(1 << t.bitWidth, (c) => _compactValue(t, c)));
  if (t == TensorType.float8e8m0) {
    if (value.isNaN || value < 0) return 255;
    if (value == 0) return 0;
    if (value.isInfinite) return saturate ? 254 : 255;
    final power = math.log(value) / math.ln2;
    final e = switch (rounding) {
      'down' => power.floor(),
      'nearest' => _roundEven(power),
      _ => power.ceil()
    };
    return (e + 127).clamp(0, saturate ? 254 : 255);
  }
  final signBit = 1 << (t.bitWidth - 1);
  final nan = entries.indexWhere((n) => n.isNaN);
  if (value.isNaN) return nan < 0 ? 0 : nan;
  final abs = value.abs();
  var top = signBit - 1;
  while (!entries[top].isFinite) top--;
  final max = entries[top], step = max - entries[top - 1];
  if (abs >= max + step / 2 && !saturate && t.bitWidth == 8) {
    if (t == TensorType.float8e5m2)
      return (value.isNegative ? signBit : 0) | (top + 1);
    return nan;
  }
  if (abs >= max) return top | (value.isNegative ? signBit : 0);
  var lo = 0, hi = top;
  while (lo < hi) {
    final mid = (lo + hi) ~/ 2;
    if (entries[mid] < abs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  var code = lo;
  if (lo > 0) {
    final up = (entries[lo] - abs).abs(), down = (abs - entries[lo - 1]).abs();
    if (down < up || (down == up && lo.isOdd)) code--;
  }
  final unsignedZero =
      t == TensorType.float8e4m3fnuz || t == TensorType.float8e5m2fnuz;
  return code |
      (value.isNegative && !(code == 0 && unsignedZero) ? signBit : 0);
}

Tensor _compactFromValues(TensorType type, List<int> shape, Iterable<num> data,
        {bool saturate = true, String rounding = 'up'}) =>
    Tensor._(
        type,
        shape,
        List<num>.unmodifiable(data.map((v) => _compactValue(type,
            _compactCode(type, v, saturate: saturate, rounding: rounding)))));
Uint8List _encodeCompact(Tensor t) {
  final bits = t.type.bitWidth, out = Uint8List((bits * t.length + 7) ~/ 8);
  for (var i = 0; i < t.length; i++) {
    final pos = i * bits ~/ 8,
        shift = i * bits % 8,
        code = _compactCode(t.type, t.values[i],
            saturate: !t.values[i].isInfinite);
    out[pos] |= (code << shift) & 255;
    if (shift + bits > 8) out[pos + 1] |= code >> (8 - shift);
  }
  return out;
}

Tensor _cast(Tensor x, TensorType dest, _Node n) {
  if (x.type == dest) return x;
  if (dest == TensorType.string)
    return Tensor.strings(x.shape, x.values.map((v) {
      if (x.type == TensorType.boolean) return v == 0 ? '0' : '1';
      if (v.isNaN) return 'NaN';
      if (v.isInfinite) return v.isNegative ? '-INF' : 'INF';
      return v.toString();
    }));
  final Iterable<num> values = x.type == TensorType.string
      ? x.stringValues.map((v) => switch (v.toLowerCase()) {
            'inf' || '+inf' => double.infinity,
            '-inf' => double.negativeInfinity,
            _ => num.parse(v)
          })
      : x.values;
  if (dest.smallInteger && x.type.floating) {
    final signed = dest.name.startsWith('int'), bits = dest.bitWidth;
    final lo = signed ? -(1 << (bits - 1)) : 0,
        hi = signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;
    return Tensor.fromValues(
        dest,
        x.shape,
        values.map((v) => v.isFinite && v.truncate() >= lo && v.truncate() <= hi
            ? v.truncate()
            : 0));
  }
  if (dest.compact)
    return _compactFromValues(dest, x.shape, values,
        saturate: n.integer('saturate', 1) != 0,
        rounding: n.attrs['round_mode'] as String? ?? 'up');
  return Tensor.fromValues(dest, x.shape, values);
}
