//! Tests for LAPACK bindings

use paste::paste;
use rlst::dense::array::DynArray;
use rlst::dense::linalg::lapack::inverse::LapackInverse;
use rlst::dense::linalg::lapack::lu::ComputedLu;
use rlst::dense::linalg::lapack::lu::LapackLu;
use rlst::dense::linalg::lapack::LapackOperations;
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

            a.r_mut().lapack().inverse().unwrap();

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

macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let dim = [8, 20];
                    let mut arr = rlst_dynamic_array!($scalar, [8, 20]);

                    arr.fill_from_seed_normally_distributed(0);

                    let lu = DynArray::new_from(&arr).lapack().lu().unwrap();


                    let l_mat = lu.get_l();
                    let u_mat = lu.get_u();
                    let p_mat = lu.get_p();

                    let res =
                        empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    rlst::assert_array_relative_eq!(res, arr, $tol)
                }

                // #[test]
                // fn [<test_lu_square_$scalar>]() {
                //     let dim = [12, 12];
                //     let mut arr = rlst_dynamic_array2!($scalar, dim);
                //
                //     arr.fill_from_seed_normally_distributed(0);
                //     let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                //     arr2.fill_from(arr.r());
                //
                //     let lu = LuDecomposition::<$scalar, _>::new(arr2).unwrap();
                //
                //     let mut l_mat = empty_array::<$scalar, 2>();
                //     let mut u_mat = empty_array::<$scalar, 2>();
                //     let mut p_mat = empty_array::<$scalar, 2>();
                //
                //     lu.get_l_resize(l_mat.r_mut());
                //     lu.get_u_resize(u_mat.r_mut());
                //     lu.get_p_resize(p_mat.r_mut());
                //
                //     let res = empty_array::<$scalar, 2>();
                //
                //     let res =
                //         res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);
                //
                //     assert_array_relative_eq!(res, arr, $tol)
                // }
                //
                // #[test]
                // fn [<test_lu_solve_$scalar>]() {
                //     let dim = [12, 12];
                //     let mut arr = rlst_dynamic_array2!($scalar, dim);
                //     arr.fill_from_seed_equally_distributed(0);
                //     let mut x_actual = rlst_dynamic_array1!($scalar, [dim[0]]);
                //     let mut rhs = rlst_dynamic_array1!($scalar, [dim[0]]);
                //     x_actual.fill_from_seed_equally_distributed(1);
                //     rhs.r_mut().simple_mult_into_resize(arr.r(), x_actual.r());
                //
                //     let lu = LuDecomposition::<$scalar,_>::new(arr).unwrap();
                //     lu.solve_vec(TransMode::NoTrans, rhs.r_mut()).unwrap();
                //
                //     assert_array_relative_eq!(x_actual, rhs, $tol)
                // }
                //
                //
                //
                // #[test]
                // fn [<test_lu_thin_$scalar>]() {
                //     let dim = [12, 8];
                //     let mut arr = rlst_dynamic_array2!($scalar, dim);
                //
                //     arr.fill_from_seed_normally_distributed(0);
                //     let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                //     arr2.fill_from(arr.r());
                //
                //     let lu = LuDecomposition::<$scalar,_>::new(arr2).unwrap();
                //
                //     let mut l_mat = empty_array::<$scalar, 2>();
                //     let mut u_mat = empty_array::<$scalar, 2>();
                //     let mut p_mat = empty_array::<$scalar, 2>();
                //
                //     lu.get_l_resize(l_mat.r_mut());
                //     lu.get_u_resize(u_mat.r_mut());
                //     lu.get_p_resize(p_mat.r_mut());
                //
                //     let res = empty_array::<$scalar, 2>();
                //
                //     let res =
                //         res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);
                //
                //     assert_array_relative_eq!(res, arr, $tol)
                // }
                //
                // #[test]
                // fn [<test_det_$scalar>]() {
                //     let dim = [2, 2];
                //     let mut arr = rlst_dynamic_array2!($scalar, dim);
                //     arr[[0, 1]] = $scalar::from_real(3.0);
                //     arr[[1, 0]] = $scalar::from_real(2.0);
                //
                //     let det = LuDecomposition::<$scalar, _>::new(arr).unwrap().det();
                //
                //     approx::assert_relative_eq!(det, $scalar::from_real(-6.0), epsilon=$tol);
                // }



            }
        };
    }

impl_lu_tests!(f64, 1E-12);
impl_lu_tests!(f32, 1E-4);
impl_lu_tests!(c64, 1E-12);
impl_lu_tests!(c32, 1E-4);
