//! Benchmark the inverse sqrt function
use std::hint::black_box;

use bytemuck;
use coe::coerce_static as to;
use criterion::{criterion_group, criterion_main, Criterion};
use pulp::{self, Scalar, Simd};
use rand::SeedableRng;
use rlst::RlstSimd;

#[cfg(target_arch = "x86_64")]
pub fn one_div_sqrt_f64(c: &mut Criterion) {
    let simd = pulp::x86::V3::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f64, _>::new(simd);

    let one = simd_for.splat(1.0);
    let vals: [f64; 8] = [1.0, 2.0, 3.0, 4.0];

    c.bench_function("1 / sqrt(x)", |b| {
        b.iter(|| {
            black_box(simd_for.div(
                black_box(one),
                black_box(simd_for.sqrt(bytemuck::cast(vals))),
            ));
        })
    });
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn one_div_sqrt_f64(c: &mut Criterion) {
    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f64, _>::new(simd);

    let one = simd_for.splat(1.0);
    let vals = [1.0, 2.0];

    c.bench_function("1 / sqrt(x)", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(simd_for.div(
                    black_box(one),
                    black_box(simd_for.sqrt(bytemuck::cast(vals))),
                ));
            }
        })
    });
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn approx_rsqrt_f64(c: &mut Criterion) {
    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f32, _>::new(simd);
    let vals: [f64; 2] = [1.0, 2.0];

    c.bench_function("approx_rsqrt", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(simd_for.approx_recip_sqrt(black_box(bytemuck::cast(vals))));
            }
        })
    });
}

criterion_group!(benches, one_div_sqrt_f64, approx_rsqrt_f64);
criterion_main!(benches);
