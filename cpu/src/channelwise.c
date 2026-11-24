#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Normalize over channels in NCL order. Processing adjacent frames together
// allows vectorization without reordering the channel reduction for any frame.
EXPORT void onnx_channel_norm(const float *x, const float *scale,
                              const float *bias, float *y, int channels,
                              int width, int first, int end, double epsilon) {
  enum { span = 8 };
  for (int start = first; start < end; start += span) {
    const int count = end - start < span ? end - start : span;
    double mean[span] = {0}, inverse[span] = {0};
    for (int c = 0; c < channels; ++c) {
      const float *row = x + (size_t)c * width + start;
      for (int t = 0; t < count; ++t) mean[t] += row[t];
    }
    for (int t = 0; t < count; ++t) mean[t] /= channels;
    for (int c = 0; c < channels; ++c) {
      const float *row = x + (size_t)c * width + start;
      for (int t = 0; t < count; ++t) {
        const double diff = row[t] - mean[t];
        inverse[t] += diff * diff;
      }
    }
    for (int t = 0; t < count; ++t)
      inverse[t] = 1.0 / sqrt(inverse[t] / channels + epsilon);
    for (int c = 0; c < channels; ++c) {
      const size_t index = (size_t)c * width + start;
      for (int t = 0; t < count; ++t)
        y[index + t] = (x[index + t] - mean[t]) * inverse[t] * scale[c]
                       + (bias ? bias[c] : 0.0);
    }
  }
}

// Valid unit-stride depthwise NCL convolution. Its ONNX Pad, if present,
// has already run. Match the portable kernel's double-precision accumulation.
EXPORT void onnx_depthwise_valid(const float *x, const float *w,
                                 const float *bias, float *y, int channels,
                                 int width, int taps, int dilation,
                                 int first, int end) {
  const int output = width - (taps - 1) * dilation;
  for (int row = first; row < end; ++row) {
    const int c = row % channels;
    const float *src = x + (size_t)row * width;
    for (int t = 0; t < output; ++t) {
      double sum = bias ? bias[c] : 0.0;
      for (int k = 0; k < taps; ++k)
        sum += (double)src[t + k * dilation] * w[(size_t)c * taps + k];
      y[(size_t)row * output + t] = (float)sum;
    }
  }
}
