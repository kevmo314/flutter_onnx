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

/// Dense CPU products for inference isolates. Apple devices use Accelerate.
class NativeCpuBackend implements MatMulBackend {
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
    if ((Platform.isAndroid || Platform.isLinux) && maxThreads > 1) {
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

  @override
  Future<void> close() async {}
}
