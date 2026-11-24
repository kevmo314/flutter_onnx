import Flutter
import Metal
import MetalPerformanceShaders

public final class FlutterOnnxAcceleratorsPlugin: NSObject, FlutterPlugin {
    private let queue = MTLCreateSystemDefaultDevice()?.makeCommandQueue()
    private let worker = DispatchQueue(label: "dev.flutteronnx.metal")

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(name: "dev.flutteronnx/matmul", binaryMessenger: registrar.messenger())
        registrar.addMethodCallDelegate(FlutterOnnxAcceleratorsPlugin(), channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let queue = queue, MPSSupportsMTLDevice(queue.device) else {
            result(nil)
            return
        }
        switch call.method {
        case "available":
            result("Metal Performance Shaders")
        case "multiply":
            guard let args = call.arguments as? [String: Any],
                  let left = args["a"] as? FlutterStandardTypedData,
                  let right = args["b"] as? FlutterStandardTypedData,
                  let m = args["m"] as? Int, let k = args["k"] as? Int, let n = args["n"] as? Int,
                  m > 0, k > 0, n > 0, max(m, max(k, n)) <= Int(Int32.max),
                  m*k <= queue.device.maxBufferLength/4, k*n <= queue.device.maxBufferLength/4,
                  m*n <= queue.device.maxBufferLength/4,
                  left.data.count == m*k*4, right.data.count == k*n*4 else {
                result(nil)
                return
            }
            worker.async {
                let output = self.multiply(left.data, right.data, m, k, n, queue)
                DispatchQueue.main.async { result(output.map { FlutterStandardTypedData(float32: $0) }) }
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func multiply(_ a: Data, _ b: Data, _ m: Int, _ k: Int, _ n: Int, _ queue: MTLCommandQueue) -> Data? {
        let device = queue.device
        func buffer(_ data: Data) -> MTLBuffer? {
            data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
                device.makeBuffer(bytes: bytes.baseAddress!, length: data.count, options: .storageModeShared)
            }
        }
        func matrix(_ buffer: MTLBuffer, _ rows: Int, _ columns: Int) -> MPSMatrix {
            let descriptor = MPSMatrixDescriptor(rows: rows, columns: columns, rowBytes: columns*4, dataType: .float32)
            return MPSMatrix(buffer: buffer, descriptor: descriptor)
        }
        guard let left = buffer(a), let right = buffer(b),
              let out = device.makeBuffer(length: m*n*4, options: .storageModeShared),
              let commands = queue.makeCommandBuffer() else { return nil }
        let product = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
            resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1, beta: 0)
        product.encode(commandBuffer: commands, leftMatrix: matrix(left,m,k),
                       rightMatrix: matrix(right,k,n), resultMatrix: matrix(out,m,n))
        commands.commit()
        commands.waitUntilCompleted()
        guard commands.status == .completed else { return nil }
        return Data(bytes: out.contents(), count: m*n*4)
    }
}
