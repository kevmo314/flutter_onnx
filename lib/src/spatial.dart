part of '../flutter_onnx.dart';

// A common NCHW window iterator for convolution and pooling.
Tensor _spatial(_Node n, List<Tensor?> args) {
  final x = args.first!;
  _require(x.shape.length == 4, 'Expected 2-D NCHW input');
  final [batch, channels, height, width] = x.shape;
  final conv = n.op == 'Conv';
  final w = conv ? args[1]! : null;
  final bias = conv ? args.elementAtOrNull(2) : null;
  final group = n.integer('group', 1);
  final outChannels =
      w == null ? channels : (w.shape.length == 4 ? w.shape[0] : -1);
  _require(
      group > 0 &&
          channels > 0 &&
          channels % group == 0 &&
          outChannels > 0 &&
          outChannels % group == 0,
      'Invalid channel groups or weight rank');
  if (w != null)
    _require(w.shape[1] == channels ~/ group, 'Weight channels mismatch');
  if (bias != null)
    _require(bias.shape.length == 1 && bias.shape[0] == outChannels,
        'Invalid Conv bias');
  final kernel = n.ints('kernel_shape') ?? w?.shape.sublist(2);
  if (kernel == null) throw ArgumentError('Missing kernel_shape');
  final strides = n.ints('strides') ?? [1, 1],
      dil = n.ints('dilations') ?? [1, 1];
  for (final pair in [kernel, strides, dil]) {
    _require(pair.length == 2 && pair.every((v) => v > 0),
        'Invalid kernel, strides or dilations');
  }
  if (w != null)
    _require(w.shape[2] == kernel[0] && w.shape[3] == kernel[1],
        'kernel_shape differs from weights');
  final pads = [...?n.ints('pads')];
  if (!n.attrs.containsKey('pads')) pads.addAll([0, 0, 0, 0]);
  _require(pads.length == 4 && pads.every((p) => p >= 0), 'Invalid pads');
  final auto = n.attrs['auto_pad'] ?? 'NOTSET';
  _require(auto == 'NOTSET' || !n.attrs.containsKey('pads'),
      'Conflicting padding modes');
  int outputDim(int input, int axis) {
    final step = strides[axis], extent = 1 + (kernel[axis] - 1) * dil[axis];
    if (auto == 'SAME_UPPER' || auto == 'SAME_LOWER') {
      final out = (input + step - 1) ~/ step;
      final padding = math.max(0, (out - 1) * step + extent - input);
      pads[axis] = (padding + (auto == 'SAME_LOWER' ? 1 : 0)) ~/ 2;
      pads[axis + 2] = padding - pads[axis];
      return out;
    }
    _require(auto == 'NOTSET' || auto == 'VALID', 'Unknown auto_pad $auto');
    final span = (input + pads[axis] + pads[axis + 2] - extent) / step;
    var out = (n.integer('ceil_mode', 0) == 0 ? span.floor() : span.ceil()) + 1;
    if (n.integer('ceil_mode', 0) != 0 &&
        (out - 1) * step >= input + pads[axis]) out--;
    return math.max(0, out);
  }

  final oh = outputDim(height, 0), ow = outputDim(width, 1);
  final kh = kernel[0], kw = kernel[1], perGroup = channels ~/ group;
  final source = x.values as Float32List;
  final weights = w?.values as Float32List?;
  final includePad = n.integer('count_include_pad', 0) != 0;
  final maxPool = n.op == 'MaxPool';
  return _float([batch, outChannels, oh, ow], (flat) {
    final ox = flat % ow,
        oy = flat ~/ ow % oh,
        oc = flat ~/ (oh * ow) % outChannels;
    final sample = flat ~/ (ow * oh * outChannels);
    final x0 = ox * strides[1] - pads[1], y0 = oy * strides[0] - pads[0];
    var acc = maxPool
        ? double.negativeInfinity
        : (bias?.values[oc].toDouble() ?? 0.0);
    var count = 0;
    final start = conv ? oc ~/ (outChannels ~/ group) * perGroup : oc;
    final depth = conv ? perGroup : 1;
    for (var c = 0; c < depth; c++) {
      for (var ky = 0; ky < kh; ky++) {
        final y = y0 + ky * dil[0];
        for (var kx = 0; kx < kw; kx++) {
          final xx = x0 + kx * dil[1];
          if (y < 0 || xx < 0 || y >= height || xx >= width) {
            if (includePad &&
                y >= -pads[0] &&
                xx >= -pads[1] &&
                y < height + pads[2] &&
                xx < width + pads[3]) count++;
            continue;
          }
          final v = source[
                  ((sample * channels + start + c) * height + y) * width + xx]
              .toDouble();
          if (weights != null) {
            acc += v * weights[((oc * depth + c) * kh + ky) * kw + kx];
          } else {
            acc = maxPool ? math.max(acc, v) : acc + v;
            count++;
          }
        }
      }
    }
    return conv || maxPool ? acc : acc / count;
  });
}
