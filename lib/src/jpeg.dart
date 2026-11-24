// The integer inverse DCT and chroma interpolation below are Dart adaptations
// of jidctint.c and jdsample.c in libjpeg-turbo 3.1.3. This software is based
// in part on the work of the Independent JPEG Group. Copyright (C) 1991-1998
// Thomas G. Lane. IDCT modifications by Guido Vollbeding (2002–2018).
// libjpeg-turbo modifications copyright (C) 2010, 2015–2016, 2020, 2022,
// 2024–2025 D. R. Commander, (C) 2009 Pierre Ossman for Cendio AB,
// (C) 2014 MIPS Technologies, Inc., (C) 2015 Google, Inc., and
// (C) 2019–2020 Arm Limited. See third_party/ijg/README.ijg.
part of '../flutter_onnx.dart';

int _descale(int value, int bits) =>
    ((value + (1 << (bits - 1))) / (1 << bits)).floor();

// One fixed-point Loeffler inverse-DCT pass. The first pass retains two
// fractional bits and the second removes the JPEG DCT normalization factor.
void _idctPass(List<int> block, int offset, int stride, int bits) {
  int at(int i) => block[offset + i * stride];
  final even0 = (at(0) + at(4)) * 8192, even1 = (at(0) - at(4)) * 8192;
  final rot = (at(2) + at(6)) * 4433;
  final even2 = rot - at(6) * 15137, even3 = rot + at(2) * 6270;
  final even = [even0 + even3, even1 + even2, even1 - even2, even0 - even3];
  final a = at(7), b = at(5), c = at(3), d = at(1);
  final shared = (a + b + c + d) * 9633;
  final ab = (a + d) * -7373, bc = (b + c) * -20995;
  final ac = shared - (a + c) * 16069, bd = shared - (b + d) * 3196;
  final odd = [
    d * 12299 + ab + bd,
    c * 25172 + bc + ac,
    b * 16819 + bc + bd,
    a * 2446 + ab + ac
  ];
  for (var i = 0; i < 4; i++) {
    block[offset + i * stride] = _descale(even[i] + odd[i], bits);
    block[offset + (7 - i) * stride] = _descale(even[i] - odd[i], bits);
  }
}

class _JpegPlane {
  final int width, height, h, v;
  final Uint8List pixels;
  _JpegPlane(this.width, this.height, this.h, this.v, this.pixels);
  factory _JpegPlane.read(img.JpegComponent component, int imageWidth,
      int imageHeight, int maxH, int maxV) {
    final h = maxH ~/ component.hSamples, v = maxV ~/ component.vSamples;
    final width = (imageWidth + h - 1) ~/ h,
        height = (imageHeight + v - 1) ~/ v;
    final pixels = Uint8List(width * height),
        quant = component.quantizationTable!;
    final block = List<int>.filled(64, 0);
    for (var by = 0; by < height; by += 8) {
      for (var bx = 0; bx < width; bx += 8) {
        final start =
            ((by ~/ 8) * component.blocksPerLineForMcu + bx ~/ 8) * 64;
        for (var k = 0; k < 64; k++)
          block[k] = component.coefficients[start + k] * quant[k];
        for (var col = 0; col < 8; col++) _idctPass(block, col, 8, 11);
        for (var row = 0; row < 8 && row + by < height; row++) {
          _idctPass(block, row * 8, 1, 18);
          for (var col = 0; col < 8 && col + bx < width; col++)
            pixels[(by + row) * width + bx + col] =
                (128 + block[row * 8 + col]).clamp(0, 255);
        }
      }
    }
    return _JpegPlane(width, height, h, v, pixels);
  }
  int sample(int x, int y) {
    final col = x ~/ h, row = y ~/ v, value = pixels[row * width + col];
    if (h == 1 && v == 1) return value;
    final neighborX = (col + (x.isOdd ? 1 : -1)).clamp(0, width - 1);
    final neighborY = (row + (y.isOdd ? 1 : -1)).clamp(0, height - 1);
    if (h == 1)
      return (3 * value +
              pixels[neighborY * width + col] +
              (y.isOdd ? 2 : 1)) ~/
          4;
    if (width <= 2) return value;
    if (v == 1)
      return (3 * value +
              pixels[row * width + neighborX] +
              (x.isOdd ? 2 : 1)) ~/
          4;
    final vertical = 3 * value + pixels[neighborY * width + col];
    final adjacent = 3 * pixels[row * width + neighborX] +
        pixels[neighborY * width + neighborX];
    return (3 * vertical + adjacent + (x.isOdd ? 7 : 8)) ~/ 16;
  }
}

img.Image _decodeJpeg(Uint8List bytes) {
  final jpeg = img.JpegData()..read(bytes), frame = jpeg.frame!;
  final keys = frame.componentsOrder;
  // Other colour spaces and sampling layouts use the image codec directly.
  if (jpeg.adobe?.transformCode == 0 ||
      frame.progressive == true ||
      frame.precision != 8 ||
      !_sameShape(keys, [1, 2, 3]) ||
      frame.components.values.any((c) =>
          frame.maxHSamples % c.hSamples != 0 ||
          frame.maxVSamples % c.vSamples != 0 ||
          frame.maxHSamples ~/ c.hSamples > 2 ||
          frame.maxVSamples ~/ c.vSamples > 2)) return jpeg.getImage();
  // The image codec releases coefficients after building its pixel planes.
  // Replay a single interleaved baseline scan into the same frame for ISLOW.
  var offset = 2;
  while (offset < bytes.length) {
    if (bytes[offset++] != 255) return jpeg.getImage();
    while (offset < bytes.length && bytes[offset] == 255) offset++;
    if (offset + 3 > bytes.length) return jpeg.getImage();
    final marker = bytes[offset++],
        length = bytes[offset] * 256 + bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) return jpeg.getImage();
    if (marker == 0xda) {
      if (length != 12 || bytes[offset + 2] != 3) return jpeg.getImage();
      final components = [
        for (var i = 0; i < 3; i++) frame.components[bytes[offset + 3 + 2 * i]]!
      ];
      frame.prepare();
      img.JpegScan(
              img.InputBuffer(bytes, bigEndian: true)..offset = offset + length,
              frame,
              components,
              jpeg.resetInterval,
              0,
              63,
              0,
              0)
          .decode();
      break;
    }
    offset += length;
  }
  if (frame.components[keys.first]!.coefficients.isEmpty)
    return jpeg.getImage();
  final w = jpeg.width!, h = jpeg.height!;
  final planes = [
    for (final id in keys)
      _JpegPlane.read(
          frame.components[id]!, w, h, frame.maxHSamples, frame.maxVSamples)
  ];
  final out = img.Image(width: w, height: h);
  for (var y = 0; y < h; y++)
    for (var x = 0; x < w; x++) {
      final lum = planes[0].sample(x, y),
          cb = planes[1].sample(x, y) - 128,
          cr = planes[2].sample(x, y) - 128;
      out.setPixelRgb(
          x,
          y,
          (lum + _descale(91881 * cr, 16)).clamp(0, 255),
          (lum + _descale(-22554 * cb - 46802 * cr, 16)).clamp(0, 255),
          (lum + _descale(116130 * cb, 16)).clamp(0, 255));
    }
  return out;
}
