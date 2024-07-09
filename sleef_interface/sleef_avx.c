#include <x86intrin.h>
#include "sleef.h"
//#include <immintrin.h>
#include <stdio.h>

typedef struct {
  double x[4];
  double y[4];
} rlst_float64x4x2;

typedef struct {
  float x[8];
  float y[8];
} rlst_float32x8x2;

typedef struct {
  float x[8];
} rlst_float32x8;

typedef struct {
  double x[4];
} rlst_float64x4;

rlst_float64x4x2 rlst_avx_sin_cos_f64(double* value) {

  __m256d simd_value = _mm256_loadu_pd(value);
  Sleef___m256d_2 simd_output = Sleef_sincosd4_u10avx2(simd_value);
  rlst_float64x4x2 output;

  _mm256_storeu_pd (output.x, simd_output.x);
  _mm256_storeu_pd (output.y, simd_output.y);
  

  return output;
}

rlst_float32x8x2 rlst_avx_sin_cos_f32(float* value) {

  __m256 simd_value = _mm256_loadu_ps(value);
  Sleef___m256_2 simd_output = Sleef_sincosf8_u10avx2(simd_value);
  rlst_float32x8x2 output;

  _mm256_storeu_ps (output.x, simd_output.x);
  _mm256_storeu_ps (output.y, simd_output.y);
  

  return output;
}

rlst_float64x4 rlst_avx_exp_f64(double* value) {

  __m256d simd_value = _mm256_loadu_pd(value);
  __m256d simd_output = Sleef_expd4_u10avx2(simd_value);
  rlst_float64x4 output;

  _mm256_storeu_pd (output.x, simd_output);
  

  return output;
}

rlst_float32x8 rlst_avx_exp_f32(float* value) {

  __m256 simd_value = _mm256_loadu_ps(value);
  __m256 simd_output = Sleef_expf8_u10avx2(simd_value);
  rlst_float32x8 output;

  _mm256_storeu_ps (output.x, simd_output);
  

  return output;
}


