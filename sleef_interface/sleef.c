#include "sleef.h"
#include <arm_neon.h>
#include <stdio.h>

typedef struct {
  double x[2];
  double y[2];
} float64x2x2;

typedef struct {
  float x[4];
  float y[4];
} float32x4x2;

float64x2x2 sin_cos_f64(double value[2] ) {

  float64x2_t simd_value = vld1q_f64(value);
  Sleef_float64x2_t_2 simd_output = Sleef_sincosd2_u10advsimd(simd_value);
  float64x2x2 output;


  vst1q_f64(output.x, simd_output.x);
  vst1q_f64(output.y, simd_output.y);
 
  return output;
}

float32x4x2 sin_cos_f32(float value[4] ) {

  float32x4_t simd_value = vld1q_f32(value);
  Sleef_float32x4_t_2 simd_output = Sleef_sincosf4_u10advsimd(simd_value);
  float32x4x2 output;


  vst1q_f32(output.x, simd_output.x);
  vst1q_f32(output.y, simd_output.y);
 
  return output;
}

