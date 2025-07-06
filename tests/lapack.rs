//! Tests for LAPACK bindings

use itertools::izip;
use num::Zero;
use paste::paste;
use rlst::assert_array_abs_diff_eq;
use rlst::dense::array::DynArray;
use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
use rlst::dense::linalg::lapack::qr::{EnablePivoting, QMode};
use rlst::dense::linalg::lapack::singular_value_decomposition::SvdMode;
use rlst::dense::linalg::lapack::symmeig::SymmEigMode;
use rlst::dot;

use rlst::*;

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
            let mut mat = rlst_dynamic_array!($scalar, [8, 5]);

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
            let mut mat = rlst_dynamic_array!($scalar, [5, 8]);

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
            let mut mat = rlst_dynamic_array!($scalar, [8, 5]);

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
            let mut mat = rlst_dynamic_array!($scalar, [5, 8]);

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
                .eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly)
                .unwrap();

            let (w2, v) = a
                .eigh(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors)
                .unwrap();

            let v = v.unwrap();

            rlst::assert_array_relative_eq!(w1, w2, $tol);

            let mut lambda = rlst_dynamic_array!($scalar, [n, n]);

            izip!(lambda.diag_iter_mut(), w1.iter_value()).for_each(|(v_elem, w_elem)| {
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

macro_rules! implement_eigendecomposition_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {
        #[test]
        fn [<test_eigendecomposition_$scalar>]() {
            let n = 11;
            let mut a = rlst_dynamic_array!($scalar, [n, n]);
            a.fill_from_seed_equally_distributed(0);

            let lam = a.eigenvalues().unwrap();

            let (t, z) = a.schur().unwrap();

            // Test the Schur decomposition

            let actual = dot!(z.r(), t.r(), z.r().conj().transpose().eval());

            rlst::assert_array_relative_eq!(actual, a, $tol);

            let (lam2, vr, vl) = a.eig(EigMode::BothEigenvectors).unwrap();

            rlst::assert_array_relative_eq!(lam, lam2, $tol);

            // Test the left eigenvectors

            // First convert a to a complex matrix

            let a_complex = a.to_type::<<$scalar as RlstScalar>::Complex>().eval();

            // Now create a diagonal matrix from the eigenvalues

            let mut diag = DynArray::from_shape([n, n]);

            izip!(diag.diag_iter_mut(), lam2.iter_value()).for_each(|(v_elem, w_elem)| {
                *v_elem = w_elem;
            });

            // Now check the left eigenvectors

            let vlh = vl.unwrap().conj().transpose().eval();

            let actual = dot!(vlh.inverse().unwrap(), dot!(diag.r(), vlh));

            // We test the absolute distance since some imaginary parts are zero
            // making relative tests fail.

            rlst::assert_array_abs_diff_eq!(actual, a_complex, $tol);

            // Now check the right eigenvectors

            let vr = vr.unwrap();

            let actual = dot!(vr.r(), dot!(diag, vr.r().inverse().unwrap()));
            rlst::assert_array_abs_diff_eq!(actual, a_complex, $tol);

        }
        }
    };
}

implement_eigendecomposition_tests!(f32, 1E-4);
implement_eigendecomposition_tests!(f64, 1E-10);
implement_eigendecomposition_tests!(c32, 5E-3);
implement_eigendecomposition_tests!(c64, 1E-10);

macro_rules! implement_svd_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {


        #[test]
        fn [<test_singular_values_$scalar>]() {
            let m = 10;
            let n = 5;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let ata = dot!(a.r().conj().transpose().eval(), a.r());

            let s = a.singularvalues().unwrap();

            let actual = ata
                .eigenvaluesh()
                .unwrap()
                .apply_unary_op(|v| <<$scalar as RlstScalar>::Real>::sqrt(v))
                .reverse_axis(0);

            rlst::assert_array_relative_eq!(s, actual, $tol);
        }

        #[test]
        fn [<test_svd_thin_compact_$scalar>]() {
            let m = 10;
            let n = 5;
            let k = std::cmp::min(m, n);
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let (s, u, vt) = a.svd(SvdMode::Compact).unwrap();

            let s = {
                let mut s_mat = DynArray::<$scalar, 2>::from_shape([k, k]);
                izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = RlstScalar::from_real(w_elem);
                });
                s_mat
            };

            let actual = dot!(u.r(), dot!(s.r(), vt.r()));
            rlst::assert_array_relative_eq!(actual, a, $tol);
        }

        #[test]
        fn [<test_svd_thin_full_$scalar>]() {
            let m = 10;
            let n = 5;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let (s, u, vt) = a.svd(SvdMode::Full).unwrap();

            let s = {
                let mut s_mat = DynArray::<$scalar, 2>::from_shape([m, n]);
                izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = RlstScalar::from_real(w_elem);
                });
                s_mat
            };

            let actual = dot!(u.r(), dot!(s.r(), vt.r()));
            rlst::assert_array_relative_eq!(actual, a, $tol);
        }

        #[test]
        fn [<test_svd_thick_compact_$scalar>]() {
            let m = 5;
            let n = 10;
            let k = std::cmp::min(m, n);
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let (s, u, vt) = a.svd(SvdMode::Compact).unwrap();

            let s = {
                let mut s_mat = DynArray::<$scalar, 2>::from_shape([k, k]);
                izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = RlstScalar::from_real(w_elem);
                });
                s_mat
            };

            let actual = dot!(u.r(), dot!(s.r(), vt.r()));
            rlst::assert_array_relative_eq!(actual, a, $tol);
        }

        #[test]
        fn [<test_svd_thick_full_$scalar>]() {
            let m = 5;
            let n = 10;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let (s, u, vt) = a.svd(SvdMode::Full).unwrap();

            let s = {
                let mut s_mat = DynArray::<$scalar, 2>::from_shape([m, n]);
                izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = RlstScalar::from_real(w_elem);
                });
                s_mat
            };

            let actual = dot!(u.r(), dot!(s.r(), vt.r()));
            rlst::assert_array_relative_eq!(actual, a, $tol);
        }


                }
    };
}

implement_svd_tests!(f32, 1E-4);
implement_svd_tests!(f64, 1E-10);
implement_svd_tests!(c32, 1E-4);
implement_svd_tests!(c64, 1E-10);

macro_rules! implement_test_solve {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        pub fn [<test_solve_square_$scalar>]() {
            let m = 5;
            let n = 5;
            let nrhs = 4;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let mut x_expected = rlst_dynamic_array!($scalar, [n, nrhs]);
            x_expected.fill_from_seed_equally_distributed(1);

            let rhs = dot!(a.r(), x_expected.r());

            let x_actual = a.solve(&rhs).unwrap();

            rlst::assert_array_relative_eq!(x_actual, x_expected, $tol);
        }

        #[test]
        pub fn [<test_solve_thin_$scalar>]() {
            let m = 10;
            let n = 5;
            let nrhs = 4;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let mut x_expected = rlst_dynamic_array!($scalar, [n, nrhs]);
            x_expected.fill_from_seed_equally_distributed(1);

            let rhs = dot!(a.r(), x_expected.r());

            let x_actual = a.solve(&rhs).unwrap();

            rlst::assert_array_relative_eq!(x_actual, x_expected, $tol);
        }

        #[test]
        pub fn [<test_solve_thick_$scalar>]() {
            let m = 5;
            let n = 10;
            let nrhs = 4;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_equally_distributed(0);

            let mut rhs = rlst_dynamic_array!($scalar, [m, nrhs]);
            rhs.fill_from_seed_equally_distributed(1);

            let x_actual = a.solve(&rhs).unwrap();

            let max_res = (dot!(a.r(), x_actual.r()) - rhs.r())
                .iter_value()
                .map(|v| v.abs())
                .fold(0.0, |acc, v| Max::max(&acc, &v));

            assert!(max_res < $tol);
        }

                }
    };
}

implement_test_solve!(f32, 1E-4);
implement_test_solve!(f64, 1E-10);
implement_test_solve!(c32, 1E-4);
implement_test_solve!(c64, 1E-10);

macro_rules! implement_triangular_solve_test {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        fn [<test_solve_triangular_$scalar>]() {
            let n = 10;
            let mut a = rlst_dynamic_array!($scalar, [n, n]);
            a.fill_from_seed_equally_distributed(0);

            for row in 0..n {
                for col in 1 + row..n {
                    a[[row, col]] = <$scalar>::zero(); // Make it lower triangular
                }
            }

            let mut x_actual = rlst_dynamic_array!($scalar, [n, 1]);
            x_actual.fill_from_seed_equally_distributed(1);

            let b = dot!(a.r(), x_actual.r());

            let x = a.solve_triangular(UpLo::Lower, &b).unwrap();

            rlst::assert_array_relative_eq!(x_actual, x, $tol);
        }


                }
    };
}

implement_triangular_solve_test!(f32, 5E-3);
implement_triangular_solve_test!(f64, 1E-10);
implement_triangular_solve_test!(c32, 5E-3);
implement_triangular_solve_test!(c64, 1E-10);

macro_rules! implement_cholesky_test {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        fn [<test_cholesky_$scalar>]() {
            let n = 10;
            let mut a = rlst_dynamic_array!($scalar, [n, n]);
            a.fill_from_seed_normally_distributed(0);

            // Make it symmetric positive definite
            a = dot!(a.r().conj().transpose().eval(), a.r());

            let z = a.cholesky(UpLo::Upper).unwrap();

            let actual = dot!(z.r().conj().transpose().eval(), z.r());

            rlst::assert_array_relative_eq!(actual, a, $tol);

            // Now solve a linear system with Cholesky

            let mut x_expected = rlst_dynamic_array!($scalar, [n, 2]);
            x_expected.fill_from_seed_equally_distributed(1);

            let b = dot!(a.r(), x_expected.r());

            let x_actual = a.cholesky_solve(UpLo::Upper, &b).unwrap();

            rlst::assert_array_relative_eq!(x_actual, x_expected, $tol);
        }

                }
    };
}

implement_cholesky_test!(f32, 1E-3);
implement_cholesky_test!(f64, 1E-10);
implement_cholesky_test!(c32, 1E-3);
implement_cholesky_test!(c64, 1E-10);

macro_rules! implement_pinv_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {

        #[test]
        fn [<test_pseudo_inverse_thin_$scalar>]() {
            let m = 20;
            let n = 10;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_normally_distributed(0);

            let pinv = a.pseudo_inverse(None, None).unwrap();

            let pinv_mat = pinv.as_matrix();

            assert_eq!(pinv_mat.shape(), [n, m]);

            let mut ident = rlst_dynamic_array!($scalar, [n, n]);
            ident.set_identity();

            let actual = dot!(pinv_mat.r(), a.r());

            rlst::assert_array_abs_diff_eq!(actual, ident, $tol);

            let mut x = rlst_dynamic_array!($scalar, [m, 2]);

            x.fill_from_seed_equally_distributed(1);

            rlst::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
        }

        #[test]
        fn [<test_pseudo_inverse_thick_$scalar>]() {
            let m = 10;
            let n = 20;
            let mut a = rlst_dynamic_array!($scalar, [m, n]);
            a.fill_from_seed_normally_distributed(0);

            let pinv = a.pseudo_inverse(None, None).unwrap();

            let pinv_mat = pinv.as_matrix();

            assert_eq!(pinv_mat.shape(), [n, m]);

            let mut ident = rlst_dynamic_array!($scalar, [m, m]);
            ident.set_identity();

            let actual = dot!(a.r(), pinv_mat.r());

            rlst::assert_array_abs_diff_eq!(actual, ident, $tol);

            let mut x = rlst_dynamic_array!($scalar, [m, 2]);

            x.fill_from_seed_equally_distributed(1);

            rlst::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
        }

                }
    };
}

implement_pinv_tests!(f32, 1E-4);
implement_pinv_tests!(f64, 1E-10);
implement_pinv_tests!(c32, 1E-4);
implement_pinv_tests!(c64, 1E-10);
