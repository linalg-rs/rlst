//! Tests for LAPACK bindings

use paste::paste;
use rlst::dense::array::DynArray;
use rlst::dense::linalg::lapack::LapackMut;
use rlst::dense::traits::SetIdentity;
use rlst::dense::types::{c32, c64};
use rlst::{assert_array_abs_diff_eq, prelude::*};

macro_rules! impl_inverse_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {
        #[test]
        fn [<test_inverse_$scalar>]() {
            let n = 100;

            let mut a = rlst_dynamic_array!($scalar, [n, n]);
            a.fill_from_seed_equally_distributed(0);
            let b = DynArray::new_from(&a);


            let mut ident = rlst_dynamic_array!($scalar, [n, n]);
            ident.set_identity();

            a.lapack_mut().inverse().unwrap();

            let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(a.r(), b.r());

            assert_array_abs_diff_eq!(actual, ident, $tol);
        }
        }
    };
}

impl_inverse_tests!(f32, 1E-5);
impl_inverse_tests!(f64, 1E-10);
impl_inverse_tests!(c32, 1E-4);
impl_inverse_tests!(c64, 1E-10);
