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

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
    Double,
  )
>(symbol: 'onnx_channel_norm', isLeaf: true)
external void _normalize(
  Pointer<Float> x,
  Pointer<Float> scale,
  Pointer<Float> bias,
  Pointer<Float> y,
  int channels,
  int width,
  int first,
  int end,
  double epsilon,
);

@Native<
  Void Function(
    Pointer<Float>,
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
>(symbol: 'onnx_depthwise_valid', isLeaf: true)
external void _depthwise(
  Pointer<Float> x,
  Pointer<Float> w,
  Pointer<Float> bias,
  Pointer<Float> y,
  int channels,
  int width,
  int taps,
  int dilation,
  int first,
  int end,
);

@Native<
  Void Function(
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_scaled_residual', isLeaf: true)
external void _scaledResidual(
  Pointer<Float> input,
  Pointer<Float> scale,
  Pointer<Float> residual,
  Pointer<Float> out,
  int channels,
  int width,
  int first,
  int end,
);

@Native<
  Pointer<Void> Function(Pointer<Float>, Pointer<Float>, Int32, Int32, Int32)
>(symbol: 'onnx_fc_create')
external Pointer<Void> _createDenseFilter(
  Pointer<Float> weights,
  Pointer<Float> bias,
  int channels,
  int filters,
  int threads,
);
@Native<Void Function(Pointer<Void>)>(symbol: 'onnx_fc_destroy')
external void _destroyDenseFilter(Pointer<Void> handle);
@Native<
  Int32 Function(
    Pointer<Void>,
    Pointer<Float>,
    Pointer<Float>,
    Int32,
    Int32,
    Int32,
    Int32,
    Int32,
  )
>(symbol: 'onnx_fc_run')
external int _runDenseFilter(
  Pointer<Void> handle,
  Pointer<Float> input,
  Pointer<Float> output,
  int batch,
  int channels,
  int width,
  int filters,
  int gelu,
);

class _DenseFilter implements Finalizable {
  static final _free = NativeFinalizer(
    Native.addressOf<NativeFunction<Void Function(Pointer<Void>)>>(
      _destroyDenseFilter,
    ),
  );
  final Pointer<Void> handle;
  final Tensor? bias;
  _DenseFilter(this.handle, this.bias, int size) {
    if (handle != nullptr) _free.attach(this, handle, externalSize: size);
  }
  static _DenseFilter create(
    Tensor weights,
    Tensor? bias,
    int threads,
  ) => using((arena) {
    final memory = arena<Float>(weights.length);
    memory.asTypedList(weights.length).setAll(0, weights.values as Float32List);
    final b = bias == null ? nullptr.cast<Float>() : arena<Float>(bias.length);
    if (bias != null)
      b.asTypedList(bias.length).setAll(0, bias.values as Float32List);
    return _DenseFilter(
      _createDenseFilter(
        memory,
        b,
        weights.shape[1],
        weights.shape[0],
        threads,
      ),
      bias,
      (weights.length + weights.shape.first) * 4,
    );
  }, malloc);
}

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
class NativeCpuBackend implements ChannelwiseBackend {
  /// Upper bound on explicit native workers for large products. Apple can
  /// process activation and bias in two independent output ranges.
  /// Accelerate manages any threads internal to its matrix operations.
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

  // Both keys are weak. Closing the backend also releases its association
  // with prepared filters while the model itself is still loaded.
  static final _denseFilters = Expando<Expando<_DenseFilter>>();
  static final _caches = Expando<Expando<_PackedFilters>>();

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
    final threads = math.max(
      1,
      math.min(maxThreads, Platform.numberOfProcessors),
    );
    _DenseFilter? dense;
    // A prepared BNNS layer reuses its weight layout across inference calls.
    // Smaller channel dimensions continue to use the packed BLAS path.
    if ((Platform.isIOS || Platform.isMacOS) &&
        filters >= 128 &&
        channels >= 128) {
      final cache = _denseFilters[this] ??= Expando<_DenseFilter>();
      dense = cache[weights];
      if (dense == null || !identical(dense.bias, bias)) {
        dense = cache[weights] = _DenseFilter.create(weights, bias, threads);
      }
    }
    return using((arena) {
      final size = batch * filters * width;
      final x = arena<Float>(input.length), y = arena<Float>(size);
      x.asTypedList(input.length).setAll(0, input.values as Float32List);
      final filter = dense;
      if (filter != null &&
          filter.handle != nullptr &&
          _runDenseFilter(
                filter.handle,
                x,
                y,
                batch,
                channels,
                width,
                filters,
                gelu ? 1 : 0,
              ) !=
              0) {
        return Float32List.fromList(y.asTypedList(size));
      }
      final filtersByWeight = _caches[this] ??= Expando<_PackedFilters>();
      var prepared = filtersByWeight[weights];
      if (prepared == null || !identical(prepared.bias, bias)) {
        prepared = filtersByWeight[weights] = _PackedFilters(weights, bias);
      }
      if (_pointwise(
            x,
            prepared.data,
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
  Float32List? scaledResidual(
    Tensor input,
    Tensor channelScale,
    Tensor residual,
  ) {
    if (input.type != TensorType.float32 ||
        channelScale.type != input.type ||
        residual.type != input.type ||
        input.shape.length != 3 ||
        channelScale.shape.length != 3 ||
        residual.shape.length != 3 ||
        input.length == 0 ||
        input.length > 0x7fffffff)
      return null;
    final [batch, channels, width] = input.shape;
    if (residual.shape[0] != batch ||
        residual.shape[1] != channels ||
        residual.shape[2] != width ||
        channelScale.shape[0] != 1 ||
        channelScale.shape[1] != channels ||
        channelScale.shape[2] != 1)
      return null;
    final x = input.values as Float32List,
        s = channelScale.values as Float32List;
    final r = residual.values as Float32List, out = Float32List(input.length);
    final tile = math.max(1, 131072 ~/ width);
    for (var row = 0; row < batch * channels; row += tile) {
      _scaledResidual(
        x.address,
        s.address,
        r.address,
        out.address,
        channels,
        width,
        row,
        math.min(row + tile, batch * channels),
      );
    }
    return out;
  }

  @override
  Float32List? normalizeChannels(
    Tensor input,
    Tensor scale,
    Tensor? bias,
    double epsilon,
  ) {
    if (input.type != TensorType.float32 || input.shape.length != 3)
      return null;
    final [batch, channels, width] = input.shape;
    bool vector(Tensor t) =>
        t.type == input.type && t.shape.length == 1 && t.length == channels;
    if (!vector(scale) ||
        (bias != null && !vector(bias)) ||
        channels == 0 ||
        input.length > 0x7fffffff)
      return null;
    final data = input.values as Float32List, out = Float32List(data.length);
    final weights = scale.values as Float32List,
        offsets = bias?.values as Float32List?;
    final frames = math.max(8, (131072 ~/ channels) & ~7);
    for (var b = 0; b < batch; b++) {
      final start = b * channels * width;
      final x = Float32List.sublistView(data, start, start + channels * width);
      final y = Float32List.sublistView(out, start, start + channels * width);
      for (var t = 0; t < width; t += frames) {
        final end = math.min(width, t + frames);
        if (offsets == null) {
          _normalize(
            x.address,
            weights.address,
            nullptr,
            y.address,
            channels,
            width,
            t,
            end,
            epsilon,
          );
        } else {
          _normalize(
            x.address,
            weights.address,
            offsets.address,
            y.address,
            channels,
            width,
            t,
            end,
            epsilon,
          );
        }
      }
    }
    return out;
  }

  @override
  Float32List? depthwiseValid(
    Tensor input,
    Tensor weights,
    Tensor? bias,
    int dilation,
  ) {
    if (input.type != TensorType.float32 ||
        weights.type != input.type ||
        input.shape.length != 3 ||
        weights.shape.length != 3 ||
        dilation <= 0)
      return null;
    final [batch, channels, width] = input.shape;
    final taps = weights.shape[2], output = width - (taps - 1) * dilation;
    if (weights.shape[0] != channels ||
        weights.shape[1] != 1 ||
        taps <= 1 ||
        output <= 0 ||
        input.length > 0x7fffffff ||
        weights.length > 0x7fffffff ||
        (bias != null &&
            (bias.type != input.type ||
                bias.shape.length != 1 ||
                bias.length != channels)))
      return null;
    final x = input.values as Float32List, w = weights.values as Float32List;
    final offsets = bias?.values as Float32List?,
        y = Float32List(batch * channels * output);
    final tile = math.max(1, 131072 ~/ (output * taps));
    for (var row = 0; row < batch * channels; row += tile) {
      final end = math.min(batch * channels, row + tile);
      if (offsets == null) {
        _depthwise(
          x.address,
          w.address,
          nullptr,
          y.address,
          channels,
          width,
          taps,
          dilation,
          row,
          end,
        );
      } else {
        _depthwise(
          x.address,
          w.address,
          offsets.address,
          y.address,
          channels,
          width,
          taps,
          dilation,
          row,
          end,
        );
      }
    }
    return y;
  }

  @override
  Future<void> close() async {
    _caches[this] = null;
    _denseFilters[this] = null;
  }
}
