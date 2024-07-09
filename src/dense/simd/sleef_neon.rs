//! Sleef Simd functions for Neon

use std::arch::aarch64::float64x2_t;

use num::Float;

use pulp::{f32x4, f64x2};

#[repr(C)]
pub struct f64x2x2(pub f64x2, pub f64x2);

#[repr(C)]
pub struct f32x4x2(pub f32x4, pub f32x4);

extern "C" {
    pub fn sin_cos_f32(a: f32x4) -> f32x4x2;
    pub fn sin_cos_f64(a: f64x2) -> f64x2x2;
}
