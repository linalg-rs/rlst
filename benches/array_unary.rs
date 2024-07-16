//! Benchmark unary array functions

use criterion::{criterion_group, criterion_main, Criterion};
use rlst::prelude::*;

const NSAMPLES: usize = 5000;

fn simd_array_sqrt(c: &mut Criterion) {
    let mut arr = rlst_dynamic_array2!(f32, [NSAMPLES, NSAMPLES], cache_aligned);
    let mut arr2 = rlst_dynamic_array2!(f32, [NSAMPLES, NSAMPLES], cache_aligned);
    arr.fill_from_seed_equally_distributed(0);

    c.bench_function("simd sqrt", |b| {
        b.iter(|| arr2.fill_from_chunked::<_, 4>(arr.view().transpose().sqrt()))
    });
}

fn array_sqrt(c: &mut Criterion) {
    let mut arr = rlst_dynamic_array2!(f32, [NSAMPLES, NSAMPLES]);
    let mut arr2 = rlst_dynamic_array2!(f32, [NSAMPLES, NSAMPLES]);
    arr.fill_from_seed_equally_distributed(0);

    c.bench_function("sqrt", |b| {
        b.iter(|| arr2.fill_from(arr.view().transpose().sqrt()))
    });
}

criterion_group!(benches_f32, simd_array_sqrt, array_sqrt);
criterion_main!(benches_f32);
