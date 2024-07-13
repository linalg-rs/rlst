//! Sleef Simd functions for AVX

use pulp::{f32x8, f64x4};

#[repr(C)]
pub struct F64x4x2(pub f64x4, pub f64x4);

#[repr(C)]
pub struct F32x8x2(pub f32x8, pub f32x8);

extern "C" {
    pub fn rlst_avx_sin_cos_f32(a: *const f32) -> F32x8x2;
    pub fn rlst_avx_sin_cos_f64(a: *const f64) -> F64x4x2;
    pub fn rlst_avx_exp_f32(a: *const f32) -> f32x8;
    pub fn rlst_avx_exp_f64(a: *const f64) -> f64x4;
}
