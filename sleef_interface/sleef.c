#include "sleef.h"
#include <arm_neon.h>
#include <stdio.h>

typedef struct {
  double x[2];
  double y[2];
} rlst_float64x2x2;

typedef struct {
  float x[4];
  float y[4];
} rlst_float32x4x2;

typedef struct {
  float x[4];
} rlst_float32x4;

typedef struct {
  double x[2];
} rlst_float64x2;

rlst_float64x2x2 rlst_neon_sin_cos_f64(double* value) {

  float64x2_t simd_value = vld1q_f64(value);
  Sleef_float64x2_t_2 simd_output = Sleef_sincosd2_u10advsimd(simd_value);
  rlst_float64x2x2 output;


  vst1q_f64(output.x, simd_output.x);
  vst1q_f64(output.y, simd_output.y);
 
  return output;
}

rlst_float32x4x2 rlst_neon_sin_cos_f32(float* value ) {

  float32x4_t simd_value = vld1q_f32(value);
  Sleef_float32x4_t_2 simd_output = Sleef_sincosf4_u10advsimd(simd_value);
  rlst_float32x4x2 output;


  vst1q_f32(output.x, simd_output.x);
  vst1q_f32(output.y, simd_output.y);
 
  return output;
}

rlst_float64x2 rlst_neon_exp_f64(double* value) {

  float64x2_t simd_value = vld1q_f64(value);
  float64x2_t simd_output = Sleef_expd2_u10advsimd(simd_value);
  rlst_float64x2 output;


  vst1q_f64(output.x, simd_output);

  
}

rlst_float64x2 rlst_neon_exp_f32(double* value) {

  float32x4_t simd_value = vld1q_f64(value);
  float32x4_t simd_output = Sleef_expf4_u10advsimd(simd_value);
  rlst_float32x4 output;


  vst1q_f64(output.x, simd_output);

  
}
