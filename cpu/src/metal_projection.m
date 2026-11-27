#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <string.h>

// Compile the shader once per device. Stringification embeds it in the native
// asset so an installed Flutter app does not need a separate resource bundle.
#define MSL(...) #__VA_ARGS__
static NSString *const projectionSource =
    @"#include <metal_stdlib>\nusing namespace metal;\n#pragma clang fp contract(off)\n"
    MSL(
      float activate(float value, device const float4 *table) {
        float x = value / 1.4142135381698608f, u = abs(x);
        if (isnan(x)) return x;
        float e = x;
        if (u >= 4) {
          e = copysign(1.0f, x);
        } else if (u != 0) {
          uint k = uint(rint(u * 1024));
          float dx = u - float(k) / 1024;
          float4 p = table[k];
          e = copysign(p.x + (p.y + dx * (p.z + dx * p.w)), x);
        }
        return (value * (1.0f + e)) * 0.5f;
      }

      kernel void project(device const float *input [[buffer(0)]],
                          device const float *weights [[buffer(1)]],
                          device float *output [[buffer(2)]],
                          constant uint4 &shape [[buffer(3)]],
                          device const float *bias [[buffer(4)]],
                          device const float4 *erfTable [[buffer(5)]],
                          uint2 tid [[thread_position_in_threadgroup]],
                          uint2 gid [[threadgroup_position_in_grid]]) {
        threadgroup float samples[16][16], filters[16][32];
        uint thread = tid.y * 8 + tid.x;
        uint row = gid.y * 16 + 2 * tid.y, col = gid.x * 32 + 4 * tid.x;
        /* Round sixteen-term partial sums before accumulating long channel reductions. */
        float4 totalA = 0, totalB = 0;
        for (uint k = 0; k < shape.z; k += 16) {
          for (uint i = thread; i < 256; i += 64) {
            uint r = i / 16, d = i % 16, source = gid.y * 16 + r;
            samples[r][d] = source < shape.x && k + d < shape.z
                ? input[source * shape.z + k + d] : 0;
          }
          for (uint i = thread; i < 512; i += 64) {
            uint f = i / 16, d = i % 16, source = gid.x * 32 + f;
            filters[d][f] = source < shape.y && k + d < shape.z
                ? weights[source * shape.z + k + d] : 0;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          float4 partA = 0, partB = 0;
          for (uint d = 0; d < 16; ++d) {
            uint j = tid.x * 4;
            float4 w(filters[d][j], filters[d][j+1], filters[d][j+2], filters[d][j+3]);
            partA = fma(w, samples[2 * tid.y][d], partA);
            partB = fma(w, samples[2 * tid.y + 1][d], partB);
          }
          totalA += partA;
          totalB += partB;
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint j = 0; j < 4 && col + j < shape.y; ++j) {
          if (row < shape.x) {
            float v = totalA[j] + bias[col+j];
            output[row * shape.y + col+j] = shape.w ? activate(v, erfTable) : v;
          }
          if (row + 1 < shape.x) {
            float v = totalB[j] + bias[col+j];
            output[(row+1) * shape.y + col+j] = shape.w ? activate(v, erfTable) : v;
          }
        }
      }
    );

@interface OnnxMetalProjection : NSObject
@property(nonatomic, strong) id<MTLBuffer> weights, bias;
@property(nonatomic) NSUInteger inputs, outputs;
@end
@implementation OnnxMetalProjection
@end

@interface OnnxMetalWorkspace : NSObject
@property(nonatomic, strong) NSMutableArray<id<MTLBuffer>> *buffers;
@end
@implementation OnnxMetalWorkspace
@end

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> product;
static id<MTLBuffer> erfTable;
static NSMutableArray<OnnxMetalWorkspace *> *workspaces;

// Metal has no float64. Store the CPU Erf approximation as a high/low split
// plus local Taylor coefficients so GELU can round its Erf output to float32.
static id<MTLBuffer> makeErfTable(void) {
  enum { entries = 4097 };
  id<MTLBuffer> buffer =
      [device newBufferWithLength:entries * 4 * sizeof(float)
                          options:MTLResourceStorageModeShared];
  if (!buffer)
    return nil;
  float *table = buffer.contents;
  for (int i = 0; i < entries; ++i) {
    const double x = i / 1024.0, t = 1 / (1 + 0.3275911 * x);
    const double p =
        (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
          0.284496736) *
             t +
         0.254829592) *
        t;
    const double e = 1 - p * exp(-x * x);
    const double derivative = 1.1283791670955126 * exp(-x * x);
    table[4 * i] = (float)e;
    table[4 * i + 1] = (float)(e - (float)e);
    table[4 * i + 2] = (float)derivative;
    table[4 * i + 3] = (float)(-x * derivative);
  }
  return buffer;
}

static BOOL prepare(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
    if (!device)
      return;
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:projectionSource
                                                  options:options
                                                    error:NULL];
    id<MTLFunction> kernel = [library newFunctionWithName:@"project"];
    if (!kernel)
      return;
    product = [device newComputePipelineStateWithFunction:kernel error:NULL];
    queue = [device newCommandQueue];
    erfTable = makeErfTable();
    workspaces = [NSMutableArray new];
  });
  return queue && product && erfTable;
}

void *onnx_metal_create(const float *weights, const float *bias, int channels,
                        int filters) {
  @autoreleasepool {
    if (!prepare())
      return NULL;
    OnnxMetalProjection *p = [OnnxMetalProjection new];
    p.inputs = channels;
    p.outputs = filters;
    p.weights =
        [device newBufferWithBytes:weights
                            length:(size_t)channels * filters * sizeof(float)
                           options:MTLResourceStorageModeShared];
    p.bias = [device newBufferWithLength:filters * sizeof(float)
                                 options:MTLResourceStorageModeShared];
    if (!p.weights || !p.bias)
      return NULL;
    if (bias)
      memcpy(p.bias.contents, bias, filters * sizeof(float));
    else
      memset(p.bias.contents, 0, filters * sizeof(float));
    return (__bridge_retained void *)p;
  }
}

void onnx_metal_destroy(void *handle) {
  @autoreleasepool {
    (void)CFBridgingRelease(handle);
  }
}

static id<MTLBuffer> temporary(OnnxMetalWorkspace *ws, NSUInteger index,
                               NSUInteger count) {
  const NSUInteger bytes = count * sizeof(float);
  id<MTLBuffer> buffer = index < ws.buffers.count ? ws.buffers[index] : nil;
  if (!buffer || buffer.length < bytes) {
    buffer = [device newBufferWithLength:bytes
                                 options:MTLResourceStorageModeShared];
    if (buffer)
      ws.buffers[index] = buffer;
  }
  return buffer;
}

static BOOL encode(id<MTLCommandBuffer> cmd, OnnxMetalProjection *p,
                   id<MTLBuffer> input, id<MTLBuffer> output, NSUInteger rows,
                   BOOL gelu) {
  id<MTLComputeCommandEncoder> pass = [cmd computeCommandEncoder];
  if (!pass)
    return NO;
  uint32_t shape[4] = {(uint32_t)rows, (uint32_t)p.outputs, (uint32_t)p.inputs,
                       gelu};
  [pass setComputePipelineState:product];
  [pass setBuffer:input offset:0 atIndex:0];
  [pass setBuffer:p.weights offset:0 atIndex:1];
  [pass setBuffer:output offset:0 atIndex:2];
  [pass setBytes:shape length:sizeof(shape) atIndex:3];
  [pass setBuffer:p.bias offset:0 atIndex:4];
  [pass setBuffer:erfTable offset:0 atIndex:5];
  [pass dispatchThreadgroups:MTLSizeMake((p.outputs + 31) / 32,
                                         (rows + 15) / 16, 1)
       threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
  [pass endEncoding];
  return YES;
}

int onnx_metal_run(void *first, void *second, const float *input, float *output,
                   int batch, int width, int gelu) {
  @autoreleasepool {
    OnnxMetalProjection *p = (__bridge OnnxMetalProjection *)first;
    OnnxMetalProjection *q = (__bridge OnnxMetalProjection *)second;
    if (!p || (q && q.inputs != p.outputs))
      return 0;
    const size_t nx = (size_t)width * p.inputs;
    const BOOL repeated =
        batch == 2 && !memcmp(input, input + nx, nx * sizeof(float));
    if (repeated)
      batch = 1;
    const NSUInteger rows = (NSUInteger)batch * width;
    OnnxMetalWorkspace *ws;
    @synchronized(workspaces) {
      ws = workspaces.lastObject;
      if (ws)
        [workspaces removeLastObject];
    }
    if (!ws) {
      ws = [OnnxMetalWorkspace new];
      ws.buffers = [NSMutableArray new];
    }
    id<MTLBuffer> x = temporary(ws, 0, rows * p.inputs);
    id<MTLBuffer> y = temporary(ws, 1, rows * p.outputs);
    id<MTLBuffer> z = q ? temporary(ws, 2, rows * q.outputs) : y;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!x || !y || !z || !cmd)
      return 0;
    float *frames = x.contents;
    for (int b = 0; b < batch; ++b)
      vDSP_mtrans(input + b * nx, 1, frames + b * nx, 1, width, p.inputs);
    if (!encode(cmd, p, x, y, rows, q || gelu) ||
        (q && !encode(cmd, q, y, z, rows, NO)))
      return 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted)
      return 0;
    const NSUInteger filters = (q ?: p).outputs;
    const size_t ny = (size_t)width * filters;
    const float *result = z.contents;
    for (int b = 0; b < batch; ++b)
      vDSP_mtrans(result + b * ny, 1, output + b * ny, 1, filters, width);
    if (repeated)
      memcpy(output + ny, output, ny * sizeof(float));
    @synchronized(workspaces) {
      if (workspaces.count < 2)
        [workspaces addObject:ws];
    }
    return 1;
  }
}
