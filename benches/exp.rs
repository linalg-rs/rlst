use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use rand::prelude::*;
use rlst::prelude::*;

struct Impl<'a, T: RlstScalar<Real = T>> {
    values: &'a [T],
}

impl<T: RlstScalar<Real = T> + RlstSimd> pulp::WithSimd for Impl<'_, T> {
    type Output = ();

    fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
        use coe::Coerce;
        let Self { values } = self;

        let (values_head, values_tail) = T::as_simd_slice::<S>(values);

        fn impl_slice<S: pulp::Simd, T: RlstScalar<Real = T> + RlstSimd>(
            simd: S,
            values: &[<T as RlstSimd>::Scalars<S>],
        ) {
            let simd = SimdFor::<T, S>::new(simd);
            for value in values.iter() {
                black_box(simd.exp(black_box(*value)));
            }
        }

        impl_slice::<S, T>(simd, values_head);
        impl_slice::<pulp::Scalar, T>(pulp::Scalar::new(), values_tail.coerce());
    }
}

macro_rules! impl_exp_bench {
    ($scalar:ty) => {
        paste::paste! {

            pub fn [<simd_exp_ $scalar>](c: &mut Criterion) {
                let nsamples = 1000;
                let mut rng = StdRng::seed_from_u64(0);

                let values = (0..nsamples).map(|_| rng.gen::<$scalar>()).collect::<Vec<_>>();

                c.bench_function(&format!("simd_exp_{}", stringify!($scalar)), |b| {
                    b.iter(|| {
                        pulp::Arch::new().dispatch(Impl::<'_, $scalar> {
                            values: values.as_slice(),
                        })
                    })
                });
            }

            pub fn [<scalar_exp_ $scalar>](c: &mut Criterion) {
                let nsamples = 1000;
                let mut rng = StdRng::seed_from_u64(0);

                let values = (0..nsamples).map(|_| rng.gen::<$scalar>()).collect::<Vec<_>>();

                c.bench_function(&format!("scalar_exp_{}", stringify!($scalar)), |b| {
                    b.iter(|| {
                        pulp::ScalarArch::new().dispatch(Impl::<'_, $scalar> {
                            values: values.as_slice(),
                        })
                    })
                });
            }




        }
    };
}

impl_exp_bench!(f32);
impl_exp_bench!(f64);

criterion_group!(benches_f32, simd_exp_f32, scalar_exp_f32);
criterion_group!(benches_f64, simd_exp_f64, scalar_exp_f64);
criterion_main!(benches_f32, benches_f64);
