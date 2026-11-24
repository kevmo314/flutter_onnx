import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter_onnx/flutter_onnx.dart';

@JS('navigator.gpu')
external JSObject? get _gpu;

JSObject _dict(Map<String, Object?> values) => values.jsify() as JSObject;

extension type _Device(JSObject _) implements JSObject {
  external _Buffer createBuffer(JSObject descriptor);
  external JSObject createShaderModule(JSObject descriptor);
  external JSPromise<JSObject> createComputePipelineAsync(JSObject descriptor);
  external JSObject createBindGroup(JSObject descriptor);
  external _Encoder createCommandEncoder();
  external _Queue get queue;
  external void destroy();
}
extension type _Queue(JSObject _) implements JSObject {
  external void writeBuffer(_Buffer buffer, int offset, JSTypedArray data);
  external void submit(JSArray<JSObject> commands);
}
extension type _Buffer(JSObject _) implements JSObject {
  external JSPromise<JSAny?> mapAsync(int mode);
  external JSArrayBuffer getMappedRange();
  external void unmap();
  external void destroy();
}
extension type _Encoder(JSObject _) implements JSObject {
  external _Pass beginComputePass();
  external void copyBufferToBuffer(
      _Buffer src, int sourceOffset, _Buffer dst, int destOffset, int size);
  external JSObject finish();
}
extension type _Pass(JSObject _) implements JSObject {
  external void setPipeline(JSObject pipeline);
  external void setBindGroup(int index, JSObject group);
  external void dispatchWorkgroups(int x, int y);
  external void end();
}

const _shader = '''
struct Shape { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<storage,read> a: array<f32>;
@group(0) @binding(1) var<storage,read> b: array<f32>;
@group(0) @binding(2) var<storage,read_write> outMatrix: array<f32>;
@group(0) @binding(3) var<uniform> s: Shape;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let col = id.x;
  let row = id.y;
  if (row >= s.m || col >= s.n) { return; }
  var sum = 0.0;
  for (var i = 0u; i < s.k; i++) {
    sum += a[row*s.k+i] * b[i*s.n+col];
  }
  outMatrix[row*s.n+col] = sum;
}
''';

/// Requests a WebGPU device and compiles the shared matrix kernel.
Future<MatMulBackend?> createMatMulBackend() async {
  final gpu = _gpu;
  if (gpu == null) return null;
  _Device? device;
  try {
    final adapter = await gpu
        .callMethod<JSPromise<JSObject?>>('requestAdapter'.toJS)
        .toDart;
    if (adapter == null) return null;
    device = _Device(await adapter
        .callMethod<JSPromise<JSObject>>('requestDevice'.toJS)
        .toDart);
    final module = device.createShaderModule(_dict({'code': _shader}));
    final pipeline = await device
        .createComputePipelineAsync(_dict({
          'layout': 'auto',
          'compute': {'module': module, 'entryPoint': 'main'}
        }))
        .toDart;
    return _WebGpu(device, pipeline);
  } catch (_) {
    device?.destroy();
    return null;
  }
}

class _WebGpu implements MatMulBackend {
  final _Device device;
  final JSObject pipeline;
  bool _closed = false;
  _WebGpu(this.device, this.pipeline);
  @override
  String get name => 'WebGPU';

  @override
  Future<Float32List?> multiply(Float32List left, Float32List right,
      {required int rows, required int inner, required int columns}) async {
    if (_closed ||
        rows <= 0 ||
        inner <= 0 ||
        columns <= 0 ||
        left.length != rows * inner ||
        right.length != inner * columns) return null;
    final bytes = rows * columns * 4;
    // Stay within baseline WebGPU binding and dispatch limits.
    if ([bytes, left.lengthInBytes, right.lengthInBytes]
            .any((s) => s > 128 * 1024 * 1024) ||
        rows > 65535 * 8 ||
        columns > 65535 * 8) return null;
    final buffers = <_Buffer>[];
    _Buffer buffer(int size, int usage) {
      final b = device.createBuffer(_dict({'size': size, 'usage': usage}));
      buffers.add(b);
      return b;
    }

    try {
      const storage = 128, uniform = 64, copyDst = 8, copySrc = 4, mapRead = 1;
      final a = buffer(left.lengthInBytes, storage | copyDst);
      final b = buffer(right.lengthInBytes, storage | copyDst);
      final c = buffer(bytes, storage | copySrc);
      final dims = buffer(16, uniform | copyDst);
      final readback = buffer(bytes, mapRead | copyDst);
      device.queue.writeBuffer(a, 0, left.toJS);
      device.queue.writeBuffer(b, 0, right.toJS);
      device.queue.writeBuffer(
          dims, 0, Uint32List.fromList([rows, inner, columns, 0]).toJS);
      final layout =
          pipeline.callMethod<JSObject>('getBindGroupLayout'.toJS, 0.toJS);
      final group = device.createBindGroup(_dict({
        'layout': layout,
        'entries': [
          for (final (index, binding) in [a, b, c, dims].indexed)
            {
              'binding': index,
              'resource': {'buffer': binding}
            }
        ]
      }));
      final encoder = device.createCommandEncoder();
      final pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups((columns + 7) ~/ 8, (rows + 7) ~/ 8);
      pass.end();
      encoder.copyBufferToBuffer(c, 0, readback, 0, bytes);
      device.queue.submit([encoder.finish()].toJS);
      await readback.mapAsync(mapRead).toDart;
      final data = Float32List.fromList(
          readback.getMappedRange().toDart.asFloat32List());
      readback.unmap();
      return data;
    } catch (_) {
      return null;
    } finally {
      for (final b in buffers) {
        b.destroy();
      }
    }
  }

  @override
  Future<void> close() async {
    if (!_closed) device.destroy();
    _closed = true;
  }
}
