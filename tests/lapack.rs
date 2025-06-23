//! Tests for LAPACK bindings

use itertools::izip;
use paste::paste;
use rlst::dense::array::DynArray;
use rlst::dense::linalg::lapack::qr::{EnablePivoting, QMode};
use rlst::dense::linalg::lapack::symmeig::SymmEigMode;
use rlst::dense::linalg::traits::Qr;
use rlst::dense::linalg::traits::{Inverse, UpLo};
use rlst::dense::linalg::traits::{Lu, SymmEig};
use rlst::dense::traits::SetIdentity;
use rlst::dense::types::TransMode;
use rlst::dense::types::{c32, c64};
use rlst::dot;
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

macro_rules! implement_qr_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        pub fn [<test_thin_qr_$scalar>]() {
            let shape = [8, 5];
            let mut mat = rlst_dynamic_array!($scalar, [8, 5]);
            let mat2 = DynArray::new_from(&mat);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = rlst_dynamic_array!($scalar, [5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::No).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            rlst::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            rlst::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thick_qr_$scalar>]() {
            let shape = [5, 8];
            let mut mat = rlst_dynamic_array!($scalar, [5, 8]);
            let mat2 = DynArray::new_from(&mat);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = rlst_dynamic_array!($scalar, [5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::No).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            rlst::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            rlst::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thin_qr_pivoted_$scalar>]() {
            let shape = [8, 5];
            let mut mat = rlst_dynamic_array!($scalar, [8, 5]);
            let mat2 = DynArray::new_from(&mat);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = rlst_dynamic_array!($scalar, [5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::Yes).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            rlst::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            rlst::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }


        #[test]
        pub fn [<test_thick_qr_pivoted_$scalar>]() {
            let shape = [5, 8];
            let mut mat = rlst_dynamic_array!($scalar, [5, 8]);
            let mat2 = DynArray::new_from(&mat);

            mat.fill_from_seed_equally_distributed(0);

            let mut ident = rlst_dynamic_array!($scalar, [5, 5]);
            ident.set_identity();

            let qr = mat.qr(EnablePivoting::Yes).unwrap();

            let q_mat = qr.q_mat(QMode::Compact).unwrap();
            let r_mat = qr.r_mat().unwrap();
            let p_t_mat = DynArray::new_from(&qr.p_mat().unwrap().transpose());

            let actual = empty_array()
                .simple_mult_into_resize(empty_array().simple_mult_into_resize(q_mat.r(), r_mat.r()), p_t_mat.r());

            rlst::assert_array_relative_eq!(actual, mat, $tol);

            let qtq = empty_array().mult_into_resize(
                TransMode::ConjTrans,
                TransMode::NoTrans,
                1.0.into(),
                q_mat.r(),
                q_mat.r(),
                1.0.into(),
            );

            rlst::assert_array_abs_diff_eq!(qtq, ident, $tol);
        }




                }
    };
}

implement_qr_tests!(f32, 1E-5);
implement_qr_tests!(f64, 1E-10);
implement_qr_tests!(c32, 1E-4);
implement_qr_tests!(c64, 1E-10);

macro_rules! implement_symm_eig_test {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        fn [<symm_eig_test_$scalar>]() {
            let n = 10;
            let mut a = rlst_dynamic_array!($scalar, [n, n]);
            a.fill_from_seed_equally_distributed(0);

            let a = DynArray::new_from(&(a.r() + a.r().conj().transpose()));

            let (w1, _) = a
                .symm_eig(UpLo::Upper, SymmEigMode::EigenvaluesOnly)
                .unwrap();

            let (w2, v) = a
                .symm_eig(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors)
                .unwrap();

            let v = v.unwrap();

            rlst::assert_array_relative_eq!(w1, w2, $tol);

            let mut lambda = rlst_dynamic_array!($scalar, [n, n]);

            izip!(lambda.diag_iter_mut(), w1.iter()).for_each(|(v_elem, w_elem)| {
                *v_elem = RlstScalar::from_real(w_elem);
            });

            let vt = DynArray::new_from(
                &v.r().conj().transpose(),
            );

            let actual = dot!(v.r(), dot!(lambda.r(), vt.r()));

            rlst::assert_array_relative_eq!(actual, a, $tol);
        }

                }
    };
}

implement_symm_eig_test!(f32, 1E-4);
implement_symm_eig_test!(f64, 1E-10);
implement_symm_eig_test!(c32, 1E-4);
implement_symm_eig_test!(c64, 1E-10);
