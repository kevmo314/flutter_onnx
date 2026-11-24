/// Device matrix kernels for flutter_onnx.
library;

export 'package:flutter_onnx/flutter_onnx.dart';
export 'src/native.dart' if (dart.library.js_interop) 'src/webgpu.dart'
    show createMatMulBackend;
