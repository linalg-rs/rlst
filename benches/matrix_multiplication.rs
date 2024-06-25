use criterion::{criterion_group, criterion_main, Criterion};

use rlst::{MultInto, TransMode};

use rand::SeedableRng;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use rlst::external::metal::AutoReleasePool;

const DIM: usize = 5000;

pub fn metal_matmul(c: &mut Criterion) {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    AutoReleasePool::execute(|| {
        let device = rlst::MetalDevice::from_default();
        let mut mat_a = rlst::rlst_metal_array2!(&device, f32, [DIM, DIM]);
        let mut mat_b = rlst::rlst_metal_array2!(&device, f32, [DIM, DIM]);
        let mut mat_c = rlst::rlst_metal_array2!(&device, f32, [DIM, DIM]);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

        mat_a.fill_from_equally_distributed(&mut rng);
        mat_b.fill_from_equally_distributed(&mut rng);
        mat_c.fill_from_equally_distributed(&mut rng);

        c.bench_function("Metal f32 matrix product", |b| {
            b.iter(|| {
                mat_c.view_mut().metal_mult_into(
                    TransMode::NoTrans,
                    TransMode::NoTrans,
                    1.0,
                    mat_a.view(),
                    mat_b.view(),
                    0.0,
                );
            })
        });
    });
}

pub fn cpu_matmul(c: &mut Criterion) {
    let mut mat_a = rlst::rlst_dynamic_array2!(f32, [DIM, DIM]);
    let mut mat_b = rlst::rlst_dynamic_array2!(f32, [DIM, DIM]);
    let mut mat_c = rlst::rlst_dynamic_array2!(f32, [DIM, DIM]);

    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);

    mat_a.fill_from_equally_distributed(&mut rng);
    mat_b.fill_from_equally_distributed(&mut rng);
    mat_c.fill_from_equally_distributed(&mut rng);

    c.bench_function("Metal f32 matrix product", |b| {
        b.iter(|| {
            mat_c.view_mut().mult_into(
                TransMode::NoTrans,
                TransMode::NoTrans,
                1.0,
                mat_a.view(),
                mat_b.view(),
                0.0,
            );
        })
    });
}

criterion_group!(benches, metal_matmul, cpu_matmul);
criterion_main!(benches);
