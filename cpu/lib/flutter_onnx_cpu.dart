import 'dart:ffi';
import 'dart:math' as math;
import 'dart:typed_data';
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

/// Dense CPU products for inference isolates. Apple devices use Accelerate.
class NativeCpuBackend implements MatMulBackend {
  const NativeCpuBackend();
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
