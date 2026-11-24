#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

// Matrices are row-major. Each call owns [first, end) output rows.
EXPORT void onnx_cpu_sgemm(const float *a, const float *b, float *c,
                          int32_t first, int32_t end, int32_t k, int32_t n) {
  a += (size_t)first * k;
  c += (size_t)first * n;
  const int32_t m = end - first;
#ifdef __APPLE__
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              1.0f, a, k, b, n, 0.0f, c, n);
#else
  // Reuse a block of the right matrix across four output rows. A fixed
  // tile width lets Clang vectorize the time axis on NEON and SSE targets.
  for (int i = 0; i < m; i += 4) {
    const int rows = m - i < 4 ? m - i : 4;
    for (int j = 0; j < n; j += 16) {
      const int cols = n - j < 16 ? n - j : 16;
      float sums[4][16] = {{0}};
      for (int p = 0; p < k; ++p) {
        const float *right = b + (size_t)p * n + j;
        float tail[16] = {0};
        if (cols != 16) {
          memcpy(tail, right, (size_t)cols * sizeof(float));
          right = tail;
        }
        for (int r = 0; r < rows; ++r) {
          const float weight = a[(size_t)(i + r) * k + p];
          for (int t = 0; t < 16; ++t) sums[r][t] += weight * right[t];
        }
      }
      for (int r = 0; r < rows; ++r)
        memcpy(c + (size_t)(i + r) * n + j, sums[r], (size_t)cols * sizeof(float));
    }
  }
#endif
}
