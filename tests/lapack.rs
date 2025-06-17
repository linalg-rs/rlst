//! Tests for LAPACK bindings

use paste::paste;
use rlst::dense::array::DynArray;
use rlst::dense::linalg::traits::Inverse;
use rlst::dense::linalg::traits::Lu;
use rlst::dense::traits::SetIdentity;
use rlst::dense::types::TransMode;
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
            let inv = a.inverse().unwrap();

            let mut ident = rlst_dynamic_array!($scalar, [n, n]);
            ident.set_identity();

            let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(inv.r(), a.r());
            assert_array_abs_diff_eq!(actual, ident, $tol);
        }
        }
    };
}

impl_inverse_tests!(f32, 1E-5);
impl_inverse_tests!(f64, 1E-10);
impl_inverse_tests!(c32, 1E-4);
impl_inverse_tests!(c64, 1E-10);

macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let mut arr = rlst_dynamic_array!($scalar, [8, 20]);

                    arr.fill_from_seed_normally_distributed(0);

                    let lu = DynArray::new_from(&arr).lu().unwrap();


                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    rlst::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_square_$scalar>]() {
                    let mut arr = rlst_dynamic_array!($scalar, [12, 12]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    rlst::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_solve_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = rlst_dynamic_array!($scalar, [12, 12]);
                    arr.fill_from_seed_equally_distributed(0);
                    let mut x_expected = rlst_dynamic_array!($scalar, [dim[0]]);
                    x_expected.fill_from_seed_equally_distributed(1);
                    let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());

                    let x_actual = arr.lu().unwrap().solve(TransMode::NoTrans, &rhs).unwrap();

                    rlst::assert_array_relative_eq!(x_actual, x_expected, $tol)
                }



                #[test]
                fn [<test_lu_thin_$scalar>]() {
                    let mut arr = rlst_dynamic_array!($scalar, [12, 8]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    rlst::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_det_$scalar>]() {
                    let mut arr = rlst_dynamic_array!($scalar, [2, 2]);
                    arr[[0, 1]] = $scalar::from_real(3.0);
                    arr[[1, 0]] = $scalar::from_real(2.0);

                    let det = arr.lu().unwrap().det();

                    approx::assert_relative_eq!(det, $scalar::from_real(-6.0), epsilon=$tol);
                }



            }
        };
    }

impl_lu_tests!(f64, 1E-12);
impl_lu_tests!(f32, 1E-4);
impl_lu_tests!(c64, 1E-12);
impl_lu_tests!(c32, 1E-4);
