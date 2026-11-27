#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <string.h>

// NCL activations are represented as [channels, batch * width] matrices. A
// fused pair uses a single command buffer and keeps the expanded data on GPU.
@interface OnnxMetalProjection : NSObject
@property(nonatomic, strong) MPSMatrix *weights;
@property(nonatomic, strong) id<MTLBuffer> bias;
@property(nonatomic) NSUInteger inputs, outputs;
@end
@implementation OnnxMetalProjection
@end

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLComputePipelineState> activation;

static BOOL prepare(void) {
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    device = MTLCreateSystemDefaultDevice();
    if (!device || !MPSSupportsMTLDevice(device)) return;
    queue = [device newCommandQueue];
    NSString *code = @"#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void activate(device float *x [[buffer(0)]], device const float *bias [[buffer(1)]], "
      "constant uint2 &shape [[buffer(2)]], uint i [[thread_position_in_grid]]) {"
      " if(i>=shape.x*shape.y) return; float v=x[i]+bias[i/shape.y];"
      " float a=v/1.4142135381698608f, t=1.0f/(1.0f+0.3275911f*abs(a));"
      " float p=(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t;"
      " float e=copysign(1.0f-p*exp(-a*a),a); x[i]=(v*(1.0f+e))*0.5f;}";
    MTLCompileOptions *options = [MTLCompileOptions new];
    options.fastMathEnabled = NO;
    id<MTLLibrary> library = [device newLibraryWithSource:code options:options error:NULL];
    id<MTLFunction> function = [library newFunctionWithName:@"activate"];
    if (function) activation = [device newComputePipelineStateWithFunction:function error:NULL];
  });
  return queue && activation;
}

static MPSMatrix *matrix(id<MTLBuffer> storage, NSUInteger rows, NSUInteger cols) {
  if (!storage) return nil;
  MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols
      rowBytes:cols*sizeof(float) dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:storage descriptor:desc];
}

void *onnx_metal_create(const float *w, const float *bias, int channels, int filters) {
  @autoreleasepool {
    if (!prepare()) return NULL;
    OnnxMetalProjection *p = [OnnxMetalProjection new];
    p.inputs = channels;
    p.outputs = filters;
    p.weights = matrix([device newBufferWithBytes:w length:(size_t)channels*filters*sizeof(float)
                        options:MTLResourceStorageModeShared], filters, channels);
    p.bias = [device newBufferWithLength:(size_t)filters*sizeof(float) options:MTLResourceStorageModeShared];
    if (!p.weights || !p.bias) return NULL;
    if (bias) memcpy(p.bias.contents, bias, filters*sizeof(float));
    else memset(p.bias.contents, 0, filters*sizeof(float));
    return (__bridge_retained void *)p;
  }
}

void onnx_metal_destroy(void *handle) {
  @autoreleasepool { (void)CFBridgingRelease(handle); }
}

static MPSMatrix *allocateMatrix(NSUInteger rows, NSUInteger cols) {
  return matrix([device newBufferWithLength:rows*cols*sizeof(float) options:MTLResourceStorageModeShared], rows, cols);
}

static BOOL encode(id<MTLCommandBuffer> cmd, OnnxMetalProjection *p,
                   MPSMatrix *x, MPSMatrix *y, BOOL gelu) {
  MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc] initWithDevice:device
      transposeLeft:NO transposeRight:NO resultRows:p.outputs resultColumns:x.columns
      interiorColumns:p.inputs alpha:1.0 beta:0.0];
  if (!mul) return NO;
  [mul encodeToCommandBuffer:cmd leftMatrix:p.weights rightMatrix:x resultMatrix:y];
  if (gelu) {
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    if (!encoder) return NO;
    uint32_t shape[2] = {(uint32_t)p.outputs, (uint32_t)y.columns};
    [encoder setComputePipelineState:activation];
    [encoder setBuffer:y.data offset:0 atIndex:0];
    [encoder setBuffer:p.bias offset:0 atIndex:1];
    [encoder setBytes:shape length:sizeof(shape) atIndex:2];
    const NSUInteger n = p.outputs*y.columns, group = activation.threadExecutionWidth;
    [encoder dispatchThreadgroups:MTLSizeMake((n+group-1)/group,1,1)
            threadsPerThreadgroup:MTLSizeMake(group,1,1)];
    [encoder endEncoding];
  }
  return YES;
}

int onnx_metal_run(void *handle, void *next, const float *input, float *output,
                   int batch, int width, int gelu) {
  @autoreleasepool {
    OnnxMetalProjection *p = (__bridge OnnxMetalProjection *)handle;
    OnnxMetalProjection *q = (__bridge OnnxMetalProjection *)next;
    if (!p || (q && q.inputs != p.outputs)) return 0;
    const int originalBatch = batch;
    const size_t inputSize = (size_t)width*p.inputs;
    if (batch == 2 && !memcmp(input,input+inputSize,inputSize*sizeof(float))) batch = 1;
    const NSUInteger cols = ((NSUInteger)batch*width + 15) & ~(NSUInteger)15;
    MPSMatrix *x = allocateMatrix(p.inputs,cols), *y = allocateMatrix(p.outputs,cols);
    MPSMatrix *z = q ? allocateMatrix(q.outputs,cols) : y;
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!x || !y || !z || !cmd) return 0;
    float *src = x.data.contents;
    for (int b=0; b<batch; ++b)
      for (NSUInteger c=0; c<p.inputs; ++c)
        memcpy(src+c*cols+b*width,input+(b*p.inputs+c)*width,width*sizeof(float));
    if (!encode(cmd,p,x,y,q || gelu) || (q && !encode(cmd,q,y,z,NO))) return 0;
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) return 0;
    OnnxMetalProjection *last = q ?: p;
    const float *values = z.data.contents, *bias = last.bias.contents;
    for (int b=0; b<batch; ++b)
      for (NSUInteger c=0; c<last.outputs; ++c) {
        float *dst = output+(b*last.outputs+c)*width;
        const float *row = values+c*cols+b*width;
        if (!q && gelu) memcpy(dst,row,width*sizeof(float));
        else for (int t=0; t<width; ++t) dst[t] = row[t]+bias[c];
      }
    if (originalBatch != batch) {
      size_t len = last.outputs*width;
      memcpy(output+len,output,len*sizeof(float));
    }
    return 1;
  }
}
