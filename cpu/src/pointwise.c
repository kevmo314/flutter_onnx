#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#elif defined(__linux__)
#include <pthread.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

enum { filters_per_tile = 16, frames_per_tile = 6, max_workers = 8 };

static inline double erf_polynomial(double x) {
  const double t = 1.0 / (1.0 + 0.3275911 * fabs(x));
  return (
      ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) *
           t +
       0.254829592) *
      t);
}

static inline float finish_gelu(float value, double x, double tail) {
  const float erf = x == 0 ? x : copysign(1.0 - tail, x);
  return (value * (1.0f + erf)) * 0.5f;
}

#ifdef __APPLE__
// Evaluate the same double-precision exponential in batches through vForce.
// Div, Erf, Add and both multiplications still round at the original
// boundaries.
static void activate_span(float *values, size_t length) {
  enum { block = 128 };
  for (size_t s = 0; s < length; s += block) {
    double x[block], exponent[block], polynomial[block];
    const int n = length - s < block ? (int)(length - s) : block;
    for (int i = 0; i < n; ++i) {
      x[i] = (float)((double)values[s + i] / 1.4142135381698608);
      exponent[i] = -x[i] * x[i];
      polynomial[i] = erf_polynomial(x[i]);
    }
    vvexp(exponent, exponent, &n);
    for (int i = 0; i < n; ++i)
      values[s + i] =
          finish_gelu(values[s + i], x[i], polynomial[i] * exponent[i]);
  }
}

typedef struct {
  const float *weights, *input, *bias;
  float *product, *output;
  int channels, filters, columns, width, batch, workers, gelu;
} apple_product;

static void apple_rows(void *context, size_t index) {
  const apple_product *job = context;
  const int tiles = job->filters / filters_per_tile;
  const int first = tiles * index / job->workers * filters_per_tile;
  const int end = tiles * (index + 1) / job->workers * filters_per_tile;
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, end - first,
              job->columns, job->channels, 1, job->weights + first,
              job->filters, job->input, job->columns, 0,
              job->product + (size_t)first * job->columns, job->columns);
  for (int b = 0; b < job->batch; ++b) {
    float *out = job->output + ((size_t)b * job->filters + first) * job->width;
    for (int f = first; f < end; ++f) {
      const float *src =
          job->product + (size_t)f * job->columns + b * job->width;
      float *dst = out + (size_t)(f - first) * job->width;
      for (int t = 0; t < job->width; ++t)
        dst[t] = src[t] + job->bias[f];
    }
    if (job->gelu)
      activate_span(out, (size_t)(end - first) * job->width);
  }
}
#else
static float expanded_gelu(float value) {
  const double x = (float)((double)value / 1.4142135381698608);
  return finish_gelu(value, x, erf_polynomial(x) * exp(-x * x));
}
#endif

// Prepare a bounded range of complete output-channel panels. Bias occupies
// the last row of every panel, so both parameter arrays share one lifetime.
EXPORT void onnx_pointwise_pack(const float *weights, const float *bias,
                                float *out, int first, int end, int channels,
                                int filters) {
#ifdef __APPLE__
  // Accelerate consumes a transposed K x F weight matrix for short NCL
  // sequences. Write adjacent filters in blocks to keep stores contiguous.
  for (int f = first; f < end; f += filters_per_tile) {
    for (int k = 0; k < channels; ++k) {
      for (int j = 0; j < filters_per_tile; ++j)
        out[(size_t)k * filters + f + j] =
            weights[(size_t)(f + j) * channels + k];
    }
  }
  float *offset = out + (size_t)filters * channels + first;
  if (bias)
    memcpy(offset, bias + first, (end - first) * sizeof(float));
  else
    memset(offset, 0, (end - first) * sizeof(float));
#else
  (void)filters;
  for (int f = first; f < end; f += filters_per_tile) {
    float *panel = out + (size_t)f * (channels + 1);
    for (int k = 0; k < channels; ++k)
      for (int j = 0; j < filters_per_tile; ++j)
        panel[k * filters_per_tile + j] =
            weights[(size_t)(f + j) * channels + k];
    for (int j = 0; j < filters_per_tile; ++j)
      panel[channels * filters_per_tile + j] = bias ? bias[f + j] : 0;
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
    const panel_vector *panel =
        (const panel_vector *)(job->weights + (size_t)f * (k + 1));
    for (int frame = 0; frame < job->frames; frame += frames_per_tile) {
      const float *x = job->samples + (size_t)frame * k;
      panel_vector sum[frames_per_tile] = {0};
      for (int d = 0; d < k; ++d) {
        const panel_vector w = panel[d];
#if defined(__clang__)
#pragma clang loop unroll(full)
#endif
        for (int t = 0; t < frames_per_tile; ++t)
          sum[t] += w * x[d * frames_per_tile + t];
      }
      for (int t = 0; t < frames_per_tile && frame + t < job->frames; ++t) {
        const int b = (frame + t) / job->width, time = (frame + t) % job->width;
        const panel_vector v = sum[t] + panel[k];
        float *dst =
            job->output + ((size_t)b * job->filters + f) * job->width + time;
        for (int j = 0; j < filters_per_tile; ++j)
          dst[(size_t)j * job->width] = job->gelu ? expanded_gelu(v[j]) : v[j];
      }
    }
  }
}

#if defined(__x86_64__)
__attribute__((target("avx"))) static void compute_avx(const product_job *job) {
  compute(job);
}
#endif
static void *run_job(void *arg) {
#if defined(__x86_64__)
  if (__builtin_cpu_supports("avx")) {
    compute_avx(arg);
    return NULL;
  }
#endif
  compute(arg);
  return NULL;
}
#endif
#endif

EXPORT int onnx_pointwise_run(const float *x, const float *weights, float *y,
                              int batch, int channels, int width, int filters,
                              int gelu, int threads) {
  // Classifier-free guidance can duplicate a latent across the batch before
  // conditioning. Only reuse a product for bit-identical input samples.
  const int repeated =
      batch == 2 && memcmp(x, x + (size_t)channels * width,
                           (size_t)channels * width * sizeof(float)) == 0;
  if (repeated)
    batch = 1;
#ifdef __APPLE__
  const float *bias = weights + (size_t)filters * channels;
  const int columns = (batch * width + 3) & ~3;
  const int direct = batch == 1 && columns == width;
  float *joined =
      direct ? NULL : calloc((size_t)channels * columns, sizeof(float));
  float *product =
      direct ? y : malloc((size_t)filters * columns * sizeof(float));
  if (!direct && (!joined || !product)) {
    free(joined);
    free(product);
    return 0;
  }
  if (!direct) {
    for (int b = 0; b < batch; ++b) {
      for (int c = 0; c < channels; ++c)
        memcpy(joined + (size_t)c * columns + b * width,
               x + ((size_t)b * channels + c) * width, width * sizeof(float));
    }
  }
  // All batches share one parameter matrix. Keep the four-frame tail padded
  // so Accelerate can select a vector kernel even for odd speech lengths.
  // Splitting a small SGEMM alone adds overhead. With GELU, independent
  // output ranges also parallelize the activation and NCL output copies.
  const int parallel = gelu && threads > 1 && filters >= 32 &&
                       (int64_t)filters * channels * columns >= 8000000;
  apple_product job = {weights,
                       direct ? x : joined,
                       bias,
                       product,
                       y,
                       channels,
                       filters,
                       columns,
                       width,
                       batch,
                       parallel ? 2 : 1,
                       gelu};
  if (parallel)
    dispatch_apply_f(job.workers,
                     dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                     &job, apple_rows);
  else
    apple_rows(&job, 0);
  free(joined);
  if (!direct)
    free(product);
#elif defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
  const int frames = batch * width;
  const size_t padded = ((size_t)frames + frames_per_tile - 1) /
                        frames_per_tile * frames_per_tile;
  float *packed = malloc(padded * channels * sizeof(float));
  if (!packed)
    return 0;
  for (int frame = 0; frame < frames; frame += frames_per_tile) {
    const float *row[frames_per_tile];
    for (int t = 0; t < frames_per_tile; ++t) {
      const int p = frame + t;
      row[t] = p < frames
                   ? x + (size_t)(p / width) * channels * width + p % width
                   : NULL;
    }
    float *dst = packed + (size_t)frame * channels;
    for (int c = 0; c < channels; ++c)
      for (int t = 0; t < frames_per_tile; ++t)
        dst[c * frames_per_tile + t] = row[t] ? row[t][(size_t)c * width] : 0;
  }
  const int tiles = filters / filters_per_tile;
  if (threads > tiles)
    threads = tiles;
  if (threads > max_workers)
    threads = max_workers;
  pthread_t workers[max_workers];
  product_job jobs[max_workers];
  int started[max_workers] = {0};
  int first = 0;
  for (int i = 0; i < threads; ++i) {
    const int end =
        (int)((int64_t)tiles * (i + 1) / threads) * filters_per_tile;
    jobs[i] = (product_job){weights, packed, y,     channels, filters,
                            width,   frames, first, end,      gelu};
    first = end;
    if (i + 1 < threads &&
        pthread_create(&workers[i], NULL, run_job, &jobs[i]) == 0)
      started[i] = 1;
    else
      run_job(&jobs[i]);
  }
  for (int i = 0; i < threads; ++i)
    if (started[i])
      pthread_join(workers[i], NULL);
  free(packed);
#else
  return 0;
#endif
  if (repeated)
    memcpy(y + (size_t)filters * width, y,
           (size_t)filters * width * sizeof(float));
  return 1;
}
