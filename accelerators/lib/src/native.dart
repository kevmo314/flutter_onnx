import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:flutter_onnx/flutter_onnx.dart';

const _channel = MethodChannel('dev.flutteronnx/matmul');

/// Returns null on platforms or devices without an available backend.
Future<MatMulBackend?> createMatMulBackend() async {
  if (defaultTargetPlatform != TargetPlatform.android &&
      defaultTargetPlatform != TargetPlatform.iOS) return null;
  try {
    final name = await _channel.invokeMethod<String>('available');
    return name == null ? null : _NativeBackend(name);
  } on MissingPluginException {
    return null;
  }
}

class _NativeBackend implements MatMulBackend {
  @override
  final String name;
  _NativeBackend(this.name);
  @override
  Future<Float32List?> multiply(Float32List left, Float32List right,
      {required int rows, required int inner, required int columns}) async {
    return _channel.invokeMethod<Float32List>('multiply',
        {'a': left, 'b': right, 'm': rows, 'k': inner, 'n': columns});
  }

  @override
  Future<void> close() async {}
}
