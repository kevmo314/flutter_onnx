#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

enum { filters_per_tile = 16, frames_per_tile = 6, max_workers = 8 };

static float expanded_gelu(float value) {
  const float divided = (float)((double)value / 1.4142135381698608);
  const double x = divided;
  double erf_value = x;
  if (x != 0) {
    const double t = 1.0 / (1.0 + 0.3275911 * fabs(x));
    const double poly = (((((1.061405429*t - 1.453152027)*t + 1.421413741)*t
                          - 0.284496736)*t + 0.254829592)*t);
    erf_value = copysign(1.0, x) * (1.0 - poly * exp(-x*x));
  }
  const float gate = 1.0f + (float)erf_value;
  return (value * gate) * 0.5f;
}

// Prepare a bounded range of complete output-channel panels. Bias occupies
// the last row of every panel, so both parameter arrays share one lifetime.
EXPORT void onnx_pointwise_pack(const float *weights, const float *bias, float *out,
                               int first, int end, int channels, int filters) {
#ifdef __APPLE__
  memcpy(out + (size_t)first * channels, weights + (size_t)first * channels,
         (size_t)(end - first) * channels * sizeof(float));
  float *offset = out + (size_t)filters * channels + first;
  if (bias) memcpy(offset, bias + first, (end - first) * sizeof(float));
  else memset(offset, 0, (end - first) * sizeof(float));
#else
  (void)filters;
  for (int f = first; f < end; f += filters_per_tile) {
    float *panel = out + (size_t)f * (channels + 1);
    for (int k = 0; k < channels; ++k)
      for (int j = 0; j < filters_per_tile; ++j)
        panel[k * filters_per_tile + j] = weights[(size_t)(f+j)*channels+k];
    for (int j = 0; j < filters_per_tile; ++j)
      panel[channels * filters_per_tile + j] = bias ? bias[f+j] : 0;
  }
#endif
}

#ifndef __APPLE__
typedef struct {
  const float *weights, *samples;
  float *output;
  int channels, filters, width, frames, first, end, gelu;
} product_job;

#if defined(__GNUC__) || defined(__clang__)
typedef float panel_vector __attribute__((vector_size(64), aligned(4)));
#define INLINE __attribute__((always_inline)) inline
static INLINE void compute(const product_job *job) {
  const int k = job->channels;
  for (int f = job->first; f < job->end; f += filters_per_tile) {
    const panel_vector *panel = (const panel_vector *)(job->weights + (size_t)f * (k+1));
    for (int frame = 0; frame < job->frames; frame += frames_per_tile) {
      const float *x = job->samples + (size_t)frame * k;
      panel_vector sum[frames_per_tile] = {0};
      for (int d = 0; d < k; ++d) {
        const panel_vector w = panel[d];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
        for (int t = 0; t < frames_per_tile; ++t) sum[t] += w * x[d*frames_per_tile+t];
      }
      for (int t = 0; t < frames_per_tile && frame+t < job->frames; ++t) {
        const int b = (frame+t) / job->width, time = (frame+t) % job->width;
        const panel_vector v = sum[t] + panel[k];
        float *dst = job->output + ((size_t)b * job->filters + f) * job->width + time;
        for (int j = 0; j < filters_per_tile; ++j)
          dst[(size_t)j * job->width] = job->gelu ? expanded_gelu(v[j]) : v[j];
      }
    }
  }
}

#if defined(__x86_64__)
__attribute__((target("avx"))) static void compute_avx(const product_job *job) { compute(job); }
#endif
static void *run_job(void *arg) {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx")) { compute_avx(arg); return NULL; }
#endif
  compute(arg);
  return NULL;
}
#endif
#endif

EXPORT int onnx_pointwise_run(const float *x, const float *weights, float *y,
                            int batch, int channels, int width, int filters, int gelu, int threads) {
  // Classifier-free guidance can duplicate a latent across the batch before
  // conditioning. Only reuse a product for bit-identical input samples.
  const int repeated = batch == 2 &&
      memcmp(x, x + (size_t)channels * width, (size_t)channels * width * sizeof(float)) == 0;
  if (repeated) batch = 1;
#ifdef __APPLE__
  const float *bias = weights + (size_t)filters * channels;
  for (int b = 0; b < batch; ++b) {
    float *out = y + (size_t)b * filters * width;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, filters, width, channels,
                1, weights, channels, x + (size_t)b * channels * width, width, 0, out, width);
    for (int f = 0; f < filters; ++f)
      for (int t = 0; t < width; ++t) {
        float value = out[(size_t)f*width+t] + bias[f];
        out[(size_t)f*width+t] = gelu ? expanded_gelu(value) : value;
      }
  }
  (void)threads;
#elif defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
  const int frames = batch * width;
  const size_t padded = ((size_t)frames + frames_per_tile-1) / frames_per_tile * frames_per_tile;
  float *packed = malloc(padded * channels * sizeof(float));
  if (!packed) return 0;
  for (int frame = 0; frame < frames; frame += frames_per_tile) {
    const float *row[frames_per_tile];
    for (int t = 0; t < frames_per_tile; ++t) {
      const int p = frame+t;
      row[t] = p < frames ? x + (size_t)(p/width)*channels*width + p%width : NULL;
    }
    float *dst = packed + (size_t)frame*channels;
    for (int c = 0; c < channels; ++c)
      for (int t = 0; t < frames_per_tile; ++t)
        dst[c*frames_per_tile+t] = row[t] ? row[t][(size_t)c*width] : 0;
  }
  const int tiles = filters / filters_per_tile;
  if (threads > tiles) threads = tiles;
  if (threads > max_workers) threads = max_workers;
  pthread_t workers[max_workers];
  product_job jobs[max_workers];
  int started[max_workers] = {0};
  int first = 0;
  for (int i = 0; i < threads; ++i) {
    const int end = (int)((int64_t)tiles * (i+1) / threads) * filters_per_tile;
    jobs[i] = (product_job){weights, packed, y, channels, filters, width, frames, first, end, gelu};
    first = end;
    if (i+1 < threads && pthread_create(&workers[i], NULL, run_job, &jobs[i]) == 0)
      started[i] = 1;
    else run_job(&jobs[i]);
  }
  for (int i = 0; i < threads; ++i) if (started[i]) pthread_join(workers[i], NULL);
  free(packed);
#else
  return 0;
#endif
  if (repeated) memcpy(y + (size_t)filters * width, y, (size_t)filters * width * sizeof(float));
  return 1;
}
