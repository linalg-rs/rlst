//! Test the accuracy of the inverse sqrt

const NSAMPLES: usize = 10000;
use rand::prelude::*;
use rlst::SimdFor;

// The allow dead code warning should not be necessary. Not sure what triggers rustc to complain if it is removed.
#[allow(dead_code)]
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn main() {
    fn rel_diff_sqrt_f32(a: f32, b: f32) -> f32 {
        let m = a.abs().max(b.abs());

        (a - b).abs() / m
    }

    fn rel_diff_sqrt_f64(a: f64, b: f64) -> f64 {
        let m = a.abs().max(b.abs());

        (a - b).abs() / m
    }

    let mut max_error_f32: f32 = 0.0;
    let mut max_error_f64: f64 = 0.0;
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..NSAMPLES {
        let sample_f32: [f32; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];
        let sample_f64: [f64; 2] = [rng.gen(), rng.gen()];

        let simd_for = SimdFor::<f32, _>::new(pulp::aarch64::Neon::try_new().unwrap());
        let res_f32 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f32));

        let simd_for = SimdFor::<f64, _>::new(pulp::aarch64::Neon::try_new().unwrap());
        let res_f64 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f64));

        let res_f32: [f32; 4] = bytemuck::cast(res_f32);
        let res_f64: [f64; 2] = bytemuck::cast(res_f64);

        max_error_f32 = max_error_f32.max(
            itertools::izip!(res_f32.iter(), sample_f32).fold(0.0, |acc, (&x_sqrt, x)| {
                acc.max(rel_diff_sqrt_f32(x_sqrt, 1.0 / x.sqrt()))
            }),
        );
        max_error_f64 = max_error_f64.max(
            itertools::izip!(res_f64.iter(), sample_f64).fold(0.0, |acc, (&x_sqrt, x)| {
                acc.max(rel_diff_sqrt_f64(x_sqrt, 1.0 / x.sqrt()))
            }),
        );
    }

    println!("Maximum relative error f32: {:.2E}", max_error_f32);
    println!("Maximum relative error f64: {:.2E}", max_error_f64);
}

#[cfg(target_arch = "x86_64")]
fn main() {
    fn rel_diff_sqrt_f32(a: f32, b: f32) -> f32 {
        let m = a.abs().max(b.abs());

        (a - b).abs() / m
    }

    fn rel_diff_sqrt_f64(a: f64, b: f64) -> f64 {
        let m = a.abs().max(b.abs());

        (a - b).abs() / m
    }

    let mut max_error_f32: f32 = 0.0;
    let mut max_error_f64: f64 = 0.0;
    let mut rng = StdRng::seed_from_u64(0);

    for _ in 0..NSAMPLES {
        let sample_f32: [f32; 8] = [
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
            rng.gen(),
        ];
        let sample_f64: [f64; 4] = [rng.gen(), rng.gen(), rng.gen(), rng.gen()];

        let simd_for = SimdFor::<f32, _>::new(pulp::x86::V3::try_new().unwrap());
        let res_f32 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f32));

        let simd_for = SimdFor::<f64, _>::new(pulp::x86::V3::try_new().unwrap());
        let res_f64 = simd_for.approx_recip_sqrt(bytemuck::cast(sample_f64));

        let res_f32: [f32; 8] = bytemuck::cast(res_f32);
        let res_f64: [f64; 4] = bytemuck::cast(res_f64);

        max_error_f32 = max_error_f32.max(
            itertools::izip!(res_f32.iter(), sample_f32).fold(0.0, |acc, (&x_sqrt, x)| {
                acc.max(rel_diff_sqrt_f32(x_sqrt, 1.0 / x.sqrt()))
            }),
        );
        max_error_f64 = max_error_f64.max(
            itertools::izip!(res_f64.iter(), sample_f64).fold(0.0, |acc, (&x_sqrt, x)| {
                acc.max(rel_diff_sqrt_f64(x_sqrt, 1.0 / x.sqrt()))
            }),
        );
    }

    println!("Maximum relative error f32: {:.2E}", max_error_f32);
    println!("Maximum relative error f64: {:.2E}", max_error_f64);
}
