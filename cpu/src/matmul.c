#include <stddef.h>
#include <stdint.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#ifndef __APPLE__
enum { panel_depth = 192, tile_width = 16 };

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define HAVE_AVX2_DISPATCH
#define INLINE __attribute__((always_inline)) inline
#else
#define INLINE inline
#endif

// Pack the right matrix once per panel. In particular, a short final row
// of a panel is zero-padded once and shared by all output filters.
static INLINE void packed_product(const float *a, const float *b, float *c,
                                  int rows, int k, int n) {
  float panel[panel_depth][tile_width];
  for (int col = 0; col < n; col += tile_width) {
    const int width = n - col < tile_width ? n - col : tile_width;
    const size_t bytes = width * sizeof(float);
    for (int base = 0; base < k; base += panel_depth) {
      const int depth = k - base < panel_depth ? k - base : panel_depth;
      for (int p = 0; p < depth; ++p) {
        if (width < tile_width) memset(panel[p], 0, sizeof(panel[p]));
        memcpy(panel[p], b + (size_t)(base + p) * n + col, bytes);
      }

      int row = 0;
      for (; row + 3 < rows; row += 4) {
        const float *left = a + (size_t)row * k + base;
        float *out = c + (size_t)row * n + col;
        float sums[4][tile_width] = {{0}};
        if (base != 0) {
          for (int r = 0; r < 4; ++r)
            memcpy(sums[r], out + (size_t)r * n, bytes);
        }
        for (int p = 0; p < depth; ++p) {
          const float x0 = left[p], x1 = left[(size_t)k + p];
          const float x2 = left[2 * (size_t)k + p], x3 = left[3 * (size_t)k + p];
          for (int t = 0; t < tile_width; ++t) {
            const float v = panel[p][t];
            sums[0][t] += x0 * v;
            sums[1][t] += x1 * v;
            sums[2][t] += x2 * v;
            sums[3][t] += x3 * v;
          }
        }
        for (int r = 0; r < 4; ++r)
          memcpy(out + (size_t)r * n, sums[r], bytes);
      }
      for (; row < rows; ++row) {
        const float *left = a + (size_t)row * k + base;
        float *out = c + (size_t)row * n + col;
        float sums[tile_width] = {0};
        if (base != 0) memcpy(sums, out, bytes);
        for (int p = 0; p < depth; ++p)
          for (int t = 0; t < tile_width; ++t)
            sums[t] += left[p] * panel[p][t];
        memcpy(out, sums, bytes);
      }
    }
  }
}

#ifdef HAVE_AVX2_DISPATCH
__attribute__((target("avx2"))) static void avx2_product(
    const float *a, const float *b, float *c, int rows, int k, int n) {
  packed_product(a, b, c, rows, k, n);
}
#endif
#endif

// Matrices are row-major. Each call owns [first, end) output rows.
EXPORT void onnx_cpu_sgemm(const float *a, const float *b, float *c,
                           int32_t first, int32_t end, int32_t k, int32_t n) {
  a += (size_t)first * k;
  c += (size_t)first * n;
  const int rows = end - first;
#ifdef __APPLE__
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, n, k,
              1.0f, a, k, b, n, 0.0f, c, n);
#else
#ifdef HAVE_AVX2_DISPATCH
  if (__builtin_cpu_supports("avx2")) {
    avx2_product(a, b, c, rows, k, n);
    return;
  }
#endif
  packed_product(a, b, c, rows, k, n);
#endif
}

#if defined(__linux__)
#include <pthread.h>

// Large products use copies in native memory so the VM can reach a safepoint
// while this non-leaf call waits for its worker threads. Output rows do not overlap.
typedef struct {
  const float *a, *b;
  float *c;
  int first, end, k, n;
} row_job;

static void *multiply_rows(void *argument) {
  const row_job *job = argument;
  onnx_cpu_sgemm(job->a, job->b, job->c, job->first, job->end, job->k, job->n);
  return NULL;
}

EXPORT void onnx_cpu_parallel_sgemm(const float *a, const float *b, float *c,
                                    int32_t rows, int32_t k, int32_t n, int32_t count) {
  enum { max_threads = 8 };
  pthread_t workers[max_threads];
  row_job jobs[max_threads];
  int started[max_threads] = {0};
  if (count > max_threads) count = max_threads;
  int first = 0;
  for (int i = 0; i < count; ++i) {
    const int end = i + 1 == count ? rows : ((int)((int64_t)rows * (i + 1) / count) & ~3);
    jobs[i] = (row_job){a, b, c, first, end, k, n};
    first = end;
    if (i + 1 < count && pthread_create(&workers[i], NULL, multiply_rows, &jobs[i]) == 0) {
      started[i] = 1;
    } else {
      multiply_rows(&jobs[i]);
    }
  }
  for (int i = 0; i < count; ++i) {
    if (started[i]) pthread_join(workers[i], NULL);
  }
}
#else
EXPORT void onnx_cpu_parallel_sgemm(const float *a, const float *b, float *c,
                                    int32_t rows, int32_t k, int32_t n, int32_t count) {
  (void)count;
  onnx_cpu_sgemm(a, b, c, 0, rows, k, n);
}
#endif
