//! Tests of array algebray operations

extern crate blas_src;

use paste::paste;
use rlst::assert_array_abs_diff_eq;
use rlst::assert_array_relative_eq;
use rlst::prelude::*;

macro_rules! impl_inverse_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

                #[test]
                fn [<test_inverse_$scalar>]() {
                    let n = 4;

                    let mut a = rlst_dynamic_array2!($scalar, [n, n]);
                    let mut b = rlst_dynamic_array2!($scalar, [n, n]);

                    let mut ident = rlst_dynamic_array2!($scalar, [n, n]);
                    ident.set_identity();

                    a.fill_from_seed_equally_distributed(0);
                    b.fill_from(a.view());

                    a.view_mut().into_inverse_alloc().unwrap();

                    let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(a.view(), b.view());

                    assert_array_abs_diff_eq!(actual, ident, $tol);
                }

            }
        };
    }

impl_inverse_tests!(f64, 1E-12);
impl_inverse_tests!(f32, 5E-6);
impl_inverse_tests!(c32, 5E-6);
impl_inverse_tests!(c64, 1E-12);

macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let dim = [8, 20];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = LuDecomposition::<$scalar,_>::new(arr2).unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_square_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = LuDecomposition::<$scalar, _>::new(arr2).unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_solve_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);
                    arr.fill_from_seed_equally_distributed(0);
                    let mut x_actual = rlst_dynamic_array1!($scalar, [dim[0]]);
                    let mut rhs = rlst_dynamic_array1!($scalar, [dim[0]]);
                    x_actual.fill_from_seed_equally_distributed(1);
                    rhs.view_mut().simple_mult_into_resize(arr.view(), x_actual.view());

                    let lu = LuDecomposition::<$scalar,_>::new(arr).unwrap();
                    lu.solve_vec(TransMode::NoTrans, rhs.view_mut()).unwrap();

                    assert_array_relative_eq!(x_actual, rhs, $tol)
                }



                #[test]
                fn [<test_lu_thin_$scalar>]() {
                    let dim = [12, 8];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = LuDecomposition::<$scalar,_>::new(arr2).unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_det_$scalar>]() {
                    let dim = [2, 2];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);
                    arr[[0, 1]] = $scalar::from_real(3.0);
                    arr[[1, 0]] = $scalar::from_real(2.0);

                    let det = LuDecomposition::<$scalar, _>::new(arr).unwrap().det();

                    approx::assert_relative_eq!(det, $scalar::from_real(-6.0), epsilon=$tol);
                }



            }
        };
    }

impl_lu_tests!(f64, 1E-12);
impl_lu_tests!(f32, 1E-5);
impl_lu_tests!(c64, 1E-12);
impl_lu_tests!(c32, 1E-5);

macro_rules! impl_pinv_tests {
    ($scalar:ty, $tol:expr) => {
        paste! {

            #[test]
            fn [<test_thick_pinv_$scalar>]() {
                let shape = [5, 10];
                let tol = 0.0;

                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                let mut pinv = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                let mut ident = rlst_dynamic_array2!($scalar, [shape[0], shape[0]]);
                ident.set_identity();

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                mat2.into_pseudo_inverse_alloc(pinv.view_mut(), tol)
                    .unwrap();

                let actual = if shape[0] >= shape[1] {
                    empty_array::<$scalar, 2>().simple_mult_into_resize(pinv, mat)
                } else {
                    empty_array::<$scalar, 2>().simple_mult_into_resize(mat, pinv)
                };

                assert_array_abs_diff_eq!(actual, ident, $tol);
            }

            #[test]
            fn [<test_thin_pinv_$scalar>]() {
                let shape = [10, 5];
                let tol = 0.0;

                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                let mut pinv = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                let mut ident = rlst_dynamic_array2!($scalar, [shape[1], shape[1]]);
                ident.set_identity();

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                mat2.into_pseudo_inverse_alloc(pinv.view_mut(), tol)
                    .unwrap();

                let actual = if shape[0] >= shape[1] {
                    empty_array::<$scalar, 2>().simple_mult_into_resize(pinv, mat)
                } else {
                    empty_array::<$scalar, 2>().simple_mult_into_resize(mat, pinv)
                };

                assert_array_abs_diff_eq!(actual, ident, $tol);
            }
        }
    };
}

impl_pinv_tests!(f32, 1E-5);
impl_pinv_tests!(f64, 1E-12);
impl_pinv_tests!(c32, 1E-5);
impl_pinv_tests!(c64, 1E-12);

macro_rules! implement_qr_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            pub fn [<test_thin_qr_$scalar>]() {
                let shape = [8, 5];
                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                let mut r_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut q_mat = rlst_dynamic_array2!($scalar, [8, 5]);
                let mut p_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut p_trans = rlst_dynamic_array2!($scalar, [5, 5]);
                let actual = rlst_dynamic_array2!($scalar, [8, 5]);
                let mut ident = rlst_dynamic_array2!($scalar, [5, 5]);
                ident.set_identity();

                let qr = QrDecomposition::<$scalar,_>::new(mat).unwrap();

                let _ = qr.get_r(r_mat.view_mut());
                let _ = qr.get_q_alloc(q_mat.view_mut());
                let _ = qr.get_p(p_mat.view_mut());

                p_trans.fill_from(p_mat.transpose());

                let actual = empty_array::<$scalar, 2>()
                    .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat.view()), p_trans);

                assert_array_relative_eq!(actual, mat2, $tol);

                let qtq = empty_array::<$scalar, 2>().mult_into_resize(
                    TransMode::ConjTrans,
                    TransMode::NoTrans,
                    1.0.into(),
                    q_mat.view(),
                    q_mat.view(),
                    1.0.into(),
                );

                assert_array_abs_diff_eq!(qtq, ident, $tol);
            }

            #[test]
            pub fn [<test_thick_qr_$scalar>]() {
                let shape = [5, 8];
                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                let mut r_mat = rlst_dynamic_array2!($scalar, [5, 8]);
                let mut q_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut p_mat = rlst_dynamic_array2!($scalar, [8, 8]);
                let mut p_trans = rlst_dynamic_array2!($scalar, [8, 8]);
                let actual = rlst_dynamic_array2!($scalar, [5, 8]);
                let mut ident = rlst_dynamic_array2!($scalar, [5, 5]);
                ident.set_identity();

                let qr = QrDecomposition::<$scalar, _>::new(mat).unwrap();

                let _ = qr.get_r(r_mat.view_mut());
                let _ = qr.get_q_alloc(q_mat.view_mut());
                let _ = qr.get_p(p_mat.view_mut());

                p_trans.fill_from(p_mat.transpose());

                let actual = empty_array::<$scalar, 2>()
                    .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat), p_trans);

                assert_array_relative_eq!(actual, mat2, $tol);

                let qtq = empty_array::<$scalar, 2>().mult_into_resize(
                    TransMode::ConjTrans,
                    TransMode::NoTrans,
                    1.0.into(),
                    q_mat.view(),
                    q_mat.view(),
                    1.0.into(),
                );

                assert_array_abs_diff_eq!(qtq, ident, $tol);
            }

                    }
        };
    }

implement_qr_tests!(f32, 1E-6);
implement_qr_tests!(f64, 1E-12);
implement_qr_tests!(c32, 1E-6);
implement_qr_tests!(c64, 1E-12);

macro_rules! impl_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

                #[test]
                fn [<test_singular_values_$scalar>]() {
                    [<test_singular_values_impl_$scalar>](5, 10, $tol);
                    [<test_singular_values_impl_$scalar>](10, 5, $tol);
                }

                #[test]
                fn [<test_svd_$scalar>]() {
                    [<test_svd_impl_$scalar>](10, 5, SvdMode::Reduced, $tol);
                    [<test_svd_impl_$scalar>](5, 10, SvdMode::Reduced, $tol);
                    [<test_svd_impl_$scalar>](10, 5, SvdMode::Full, $tol);
                    [<test_svd_impl_$scalar>](5, 10, SvdMode::Full, $tol);
                }

                fn [<test_singular_values_impl_$scalar>](m: usize, n: usize, tol: <$scalar as RlstScalar>::Real) {
                    let k = std::cmp::min(m, n);
                    let mut mat = rlst_dynamic_array2!($scalar, [m, m]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, m]);
                    let mut sigma = rlst_dynamic_array2!($scalar, [m, n]);

                    mat.fill_from_seed_equally_distributed(0);
                    let qr = QrDecomposition::<$scalar,_>::new(mat).unwrap();
                    qr.get_q_alloc(q.view_mut()).unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = ((k - index) as <$scalar as RlstScalar>::Real).into();
                    }

                    let a = empty_array::<$scalar, 2>().simple_mult_into_resize(q.view(), sigma.view());

                    let mut singvals = rlst_dynamic_array1!(<$scalar as RlstScalar>::Real, [k]);

                    a.into_singular_values_alloc(singvals.data_mut()).unwrap();

                    for index in 0..k {
                        approx::assert_relative_eq!(singvals[[index]], sigma[[index, index]].re(), epsilon = tol);
                    }
                }

                fn [<test_svd_impl_$scalar>](m: usize, n: usize, mode: SvdMode, tol: <$scalar as RlstScalar>::Real) {
                    let k = std::cmp::min(m, n);

                    let mut mat_u;
                    let mut u;
                    let mut mat_vt;
                    let mut vt;
                    let mut sigma;

                    match mode {
                        SvdMode::Full => {
                            mat_u = rlst_dynamic_array2!($scalar, [m, m]);
                            u = rlst_dynamic_array2!($scalar, [m, m]);
                            mat_vt = rlst_dynamic_array2!($scalar, [n, n]);
                            vt = rlst_dynamic_array2!($scalar, [n, n]);
                            sigma = rlst_dynamic_array2!($scalar, [m, n]);
                        }
                        SvdMode::Reduced => {
                            mat_u = rlst_dynamic_array2!($scalar, [m, k]);
                            u = rlst_dynamic_array2!($scalar, [m, k]);
                            mat_vt = rlst_dynamic_array2!($scalar, [k, n]);
                            vt = rlst_dynamic_array2!($scalar, [k, n]);
                            sigma = rlst_dynamic_array2!($scalar, [k, k]);
                        }
                    }

                    mat_u.fill_from_seed_equally_distributed(0);
                    mat_vt.fill_from_seed_equally_distributed(1);

                    let qr = QrDecomposition::<$scalar,_>::new(mat_u).unwrap();
                    qr.get_q_alloc(u.view_mut()).unwrap();

                    let qr = QrDecomposition::<$scalar,_>::new(mat_vt).unwrap();
                    qr.get_q_alloc(vt.view_mut()).unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = ((k - index) as <$scalar as RlstScalar>::Real).into();
                    }

                    let a = empty_array::<$scalar, 2>().simple_mult_into_resize(
                        empty_array::<$scalar, 2>().simple_mult_into_resize(u.view(), sigma.view()),
                        vt.view(),
                    );

                    let mut expected = rlst_dynamic_array2!($scalar, a.shape());
                    expected.fill_from(a.view());

                    u.set_zero();
                    vt.set_zero();
                    sigma.set_zero();

                    let mut singvals = rlst_dynamic_array1!(<$scalar as RlstScalar>::Real, [k]);

                    a.into_svd_alloc(u.view_mut(), vt.view_mut(), singvals.data_mut(), mode)
                        .unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = singvals[[index]].into();
                    }

                    let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(
                        empty_array::<$scalar, 2>().simple_mult_into_resize(u, sigma),
                        vt,
                    );

                    assert_array_relative_eq!(expected, actual, tol);
                }


            }
        };
    }

impl_tests!(f32, 1E-5);
impl_tests!(f64, 1E-12);
impl_tests!(c32, 1E-5);
impl_tests!(c64, 1E-12);
