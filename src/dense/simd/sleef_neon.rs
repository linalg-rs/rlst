//! Sleef Simd functions for Neon

use pulp::{f32x4, f64x2};

#[repr(C)]
pub struct f64x2x2(pub f64x2, pub f64x2);

#[repr(C)]
pub struct f32x4x2(pub f32x4, pub f32x4);

extern "C" {
    pub fn rlst_neon_sin_cos_f32(a: *const f32) -> f32x4x2;
    pub fn rlst_neon_sin_cos_f64(a: *const f64) -> f64x2x2;
    pub fn rlst_neon_exp_f32(a: *const f32) -> f32x4;
    pub fn rlst_neon_exp_f64(a: *const f64) -> f64x2;
}
