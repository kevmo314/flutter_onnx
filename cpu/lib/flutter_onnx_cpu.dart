import 'dart:ffi';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import 'package:flutter_onnx/flutter_onnx.dart';

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_cpu_sgemm', isLeaf: true)
external void _multiply(
  Pointer<Float> a,
  Pointer<Float> b,
  Pointer<Float> c,
  int first,
  int end,
  int k,
  int n,
);

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_cpu_parallel_sgemm')
external void _parallelMultiply(
  Pointer<Float> left,
  Pointer<Float> right,
  Pointer<Float> out,
  int rows,
  int inner,
  int columns,
  int threads,
);

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_pointwise_pack', isLeaf: true)
external void _pack(
  Pointer<Float> weights,
  Pointer<Float> bias,
  Pointer<Float> packed,
  int first,
  int end,
  int channels,
  int filters,
);

@Native<
  Int32 Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_pointwise_run')
external int _pointwise(
  Pointer<Float> input,
  Pointer<Float> weights,
  Pointer<Float> result,
  int batch,
  int channels,
  int width,
  int filters,
  int gelu,
  int threads,
);

class _PackedFilters implements Finalizable {
  static final _free = NativeFinalizer(malloc.nativeFree);
  final Pointer<Float> data;
  final Tensor? bias;
  _PackedFilters(Tensor weights, this.bias)
    : data = malloc<Float>(weights.length + weights.shape.first) {
    _free.attach(
      this,
      data.cast(),
      externalSize: (weights.length + weights.shape.first) * 4,
    );
    final filters = weights.shape[0], channels = weights.shape[1];
    final w = weights.values as Float32List, b = bias?.values as Float32List?;
    final tile = math.max(16, (262144 ~/ channels) & ~15);
    for (var f = 0; f < filters; f += tile) {
      if (b == null) {
        _pack(
          w.address,
          nullptr,
          data,
          f,
          math.min(f + tile, filters),
          channels,
          filters,
        );
      } else {
        _pack(
          w.address,
          b.address,
          data,
          f,
          math.min(f + tile, filters),
          channels,
          filters,
        );
      }
    }
  }
}

/// Dense CPU products for inference isolates. Apple devices use Accelerate.
class NativeCpuBackend implements PointwiseConvBackend {
  /// Upper bound on native workers for large Android and Linux products.
  /// Apple Accelerate schedules its own workers.
  final int maxThreads;
  const NativeCpuBackend({this.maxThreads = 2}) : assert(maxThreads > 0);
  @override
  String get name => 'Native CPU';
  @override
  Future<Float32List?> multiply(
    Float32List left,
    Float32List right, {
    required int rows,
    required int inner,
    required int columns,
  }) async {
    if (rows < 0 ||
        inner < 0 ||
        columns < 0 ||
        rows > 0x7fffffff ||
        inner > 0x7fffffff ||
        columns > 0x7fffffff ||
        left.length != rows * inner ||
        right.length != inner * columns) {
      throw ArgumentError('Invalid matrix dimensions');
    }
    final result = Float32List(rows * columns);
    if (result.isEmpty || inner == 0) return result;
    // Keep working memory bounded for large embedding and projection tables.
    final copyBytes =
        left.lengthInBytes + right.lengthInBytes + result.lengthInBytes;
    if ((Platform.isAndroid || Platform.isLinux) &&
        maxThreads > 1 &&
        copyBytes <= 64 * 1024 * 1024) {
      final threads = [
        maxThreads, Platform.numberOfProcessors, rows ~/ 4,
        // Smaller products are faster without copying or starting workers.
        (rows * inner * columns) ~/ 8000000,
      ].reduce(math.min);
      if (threads > 1) {
        using((arena) {
          final a = arena<Float>(left.length), b = arena<Float>(right.length);
          final c = arena<Float>(result.length);
          a.asTypedList(left.length).setAll(0, left);
          b.asTypedList(right.length).setAll(0, right);
          _parallelMultiply(
            a,
            b,
            c,
            rows,
            inner,
            columns,
            math.min(8, threads),
          );
          result.setAll(0, c.asTypedList(result.length));
        }, malloc);
        return result;
      }
    }
    // Bound the time spent in a single leaf call so the VM can reach a
    // safepoint between tiles. No native code retains these pointers.
    final tile = math.max(4, (4000000 ~/ (inner * columns)) & ~3);
    for (var row = 0; row < rows; row += tile) {
      _multiply(
        left.address,
        right.address,
        result.address,
        row,
        math.min(row + tile, rows),
        inner,
        columns,
      );
    }
    return result;
  }

  static final _filters = Expando<_PackedFilters>();

  @override
  Future<Float32List?> pointwise(
    Tensor input,
    Tensor weights,
    Tensor? bias, {
    required bool gelu,
  }) async {
    if (!(Platform.isLinux ||
            Platform.isAndroid ||
            Platform.isMacOS ||
            Platform.isIOS) ||
        input.type != TensorType.float32 ||
        weights.type != input.type ||
        input.shape.length != 3 ||
        weights.shape.length != 3 ||
        weights.shape.last != 1)
      return null;
    final batch = input.shape[0],
        channels = input.shape[1],
        width = input.shape[2],
        filters = weights.shape[0];
    if (weights.shape[1] != channels ||
        filters % 16 != 0 ||
        filters == 0 ||
        channels == 0 ||
        batch == 0 ||
        width == 0 ||
        (bias != null &&
            (bias.type != input.type ||
                bias.shape.length != 1 ||
                bias.length != filters)) ||
        batch * filters * width > 16 * 1024 * 1024 ||
        input.length > 16 * 1024 * 1024 ||
        weights.length > 16 * 1024 * 1024 ||
        weights.length * batch * width < 250000)
      return null;
    var cached = _filters[weights];
    if (cached == null || !identical(cached.bias, bias)) {
      cached = _filters[weights] = _PackedFilters(weights, bias);
    }
    final parameters = cached;
    return using((arena) {
      final size = batch * filters * width;
      final x = arena<Float>(input.length), y = arena<Float>(size);
      x.asTypedList(input.length).setAll(0, input.values as Float32List);
      final threads = math.max(
        1,
        math.min(maxThreads, Platform.numberOfProcessors),
      );
      if (_pointwise(
            x,
            parameters.data,
            y,
            batch,
            channels,
            width,
            filters,
            gelu ? 1 : 0,
            threads,
          ) ==
          0)
        return null;
      return Float32List.fromList(y.asTypedList(size));
    }, malloc);
  }

  @override
  Future<void> close() async {}
}
