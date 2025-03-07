//! Benchmark the inverse sqrt function
use std::hint::black_box;

#[cfg(target_arch = "x86_64")]
use bytemuck;
use criterion::{criterion_group, criterion_main, Criterion};

#[cfg(target_arch = "x86_64")]
pub fn one_div_sqrt_f32(c: &mut Criterion) {
    let simd = pulp::x86::V3::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f32, _>::new(simd);

    let one = simd_for.splat(1.0);
    let vals: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

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

#[cfg(target_arch = "x86_64")]
pub fn approx_rsqrt_f32(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    let simd = pulp::x86::V3::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f32, _>::new(simd);

    let vals: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    c.bench_function("approx_rsqrt", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(simd_for.approx_recip_sqrt(black_box(bytemuck::cast(vals))));
            }
        })
    });
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub fn one_div_sqrt_f32(c: &mut Criterion) {
    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f32, _>::new(simd);

    let one = simd_for.splat(1.0);
    let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

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
pub fn approx_rsqrt_f32(c: &mut Criterion) {
    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f32, _>::new(simd);
    let vals: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    c.bench_function("approx_rsqrt", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                black_box(simd_for.approx_recip_sqrt(black_box(bytemuck::cast(vals))));
            }
        })
    });
}

criterion_group!(benches, approx_rsqrt_f32, one_div_sqrt_f32);
criterion_main!(benches);
