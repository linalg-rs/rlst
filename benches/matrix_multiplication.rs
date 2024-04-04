use criterion::{criterion_group, criterion_main, Criterion};

use rlst::dense::batched_gemm::{BatchedGemm, DefaultCpuBatchedGemm};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rlst::external::metal::{batched_gemm::MetalBatchedGemm, AutoReleasePool};

const DIM: usize = 5000;

extern crate blas_src;
extern crate lapack_src;

pub fn metal_matmul(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    AutoReleasePool::execute(|| {
        let number_of_matrices = 1;
        let mut batched_gemm =
            MetalBatchedGemm::new((DIM, DIM), (DIM, DIM), number_of_matrices, 1.0, 0.0);

        for index in 0..number_of_matrices {
            let mut left_matrix = batched_gemm.left_matrix_mut(index).unwrap();
            left_matrix.fill_from_seed_equally_distributed(0);
            let mut right_matrix = batched_gemm.right_matrix_mut(index).unwrap();
            right_matrix.fill_from_seed_equally_distributed(1);
        }

        c.bench_function("Metal f32 matrix product", |b| {
            b.iter(|| batched_gemm.evaluate().unwrap())
        });
    });
}

pub fn cpu_matmul(c: &mut Criterion) {
    let mut batched_gemm = DefaultCpuBatchedGemm::<f32>::new((DIM, DIM), (DIM, DIM), 1, 1.0, 0.0);

    let mut left_matrix = batched_gemm.left_matrix_mut(0).unwrap();
    left_matrix.fill_from_seed_equally_distributed(0);
    let mut right_matrix = batched_gemm.right_matrix_mut(0).unwrap();
    right_matrix.fill_from_seed_equally_distributed(1);

    c.bench_function("CPU f32 matrix product", |b| {
        b.iter(|| batched_gemm.evaluate().unwrap())
    });
}

criterion_group!(benches, metal_matmul, cpu_matmul);
criterion_main!(benches);
