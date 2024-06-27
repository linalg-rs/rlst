//! Benchmark the inverse sqrt function
use std::hint::black_box;

use bytemuck;
use coe::coerce_static as to;
use criterion::{criterion_group, criterion_main, Criterion};
use pulp::{self, Scalar, Simd};
use rand::SeedableRng;
use rlst::RlstSimd;

pub fn one_div_sqrt_f64(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    let simd = pulp::x86::V3::try_new().unwrap();
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let simd = pulp::aarch64::Neon::try_new().unwrap();

    let simd_for = rlst::dense::simd::SimdFor::<f64, _>::new(simd);

    let one = simd_for.splat(1.0);
    let vals = [1.0, 2.0, 3.0, 4.0];

    c.bench_function("1 / sqrt(x)", |b| {
        b.iter(|| {
            black_box(simd_for.div(
                black_box(one),
                black_box(simd_for.sqrt(bytemuck::cast(vals))),
            ));
        })
    });
}

criterion_group!(benches, one_div_sqrt_f64);
criterion_main!(benches);
