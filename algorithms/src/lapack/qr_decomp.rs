use crate::linalg::LinAlg;
use crate::traits::qr_decomp_trait::QRTrait;
use crate::traits::trisolve_trait::Trisolve;
use crate::{lapack::LapackData, traits::qr_decomp_trait::QRDecomposableTrait};
use lapacke;
use num::{One, Zero};
use rlst_common::traits::Copy;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_mat, MatrixD};
use rlst_dense::{RawAccess, RawAccessMut, Shape, Stride};

use super::{
    check_lapack_stride, DenseMatrixLinAlgBuilder, SideMode, TransposeMode, TriangularDiagonal,
    TriangularType,
};

pub struct QRDecompLapack<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
    data: LapackData<T, Mat>,
    tau: Vec<T>,
}

macro_rules! qr_decomp_impl {
    ($scalar:ty, $lapack_qr:ident, $lapack_qr_solve:ident, $lapack_qmut:ident, $trans:ident) => {
        impl<'a, Mat: Copy> QRDecomposableTrait for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar> + RawAccess<T = $scalar> + Shape + Stride,
        {
            type T = $scalar;
            type Out = QRDecompLapack<$scalar, <Mat as Copy>::Out>;
            /// Returns the QR decomposition of the input matrix assuming full rank and using LAPACK xGEQRF
            fn qr(self) -> RlstResult<QRDecompLapack<$scalar, <Mat as Copy>::Out>> {
                let mut copied = self.into_lapack()?;
                let dim = copied.mat.shape();
                let stride = copied.mat.stride();

                let m = dim.0 as i32;
                let n = dim.1 as i32;
                let lda = stride.1 as i32;
                let mut tau: Vec<$scalar> =
                    vec![<$scalar as Zero>::zero(); std::cmp::min(dim.0, dim.1)];

                let info = unsafe {
                    lapacke::$lapack_qr(
                        lapacke::Layout::ColumnMajor,
                        m,
                        n,
                        copied.mat.data_mut(),
                        lda,
                        &mut tau,
                    )
                };

                if info == 0 {
                    return Ok(QRDecompLapack { data: copied, tau });
                } else {
                    return Err(RlstError::LapackError(info));
                }
            }

            fn qr_and_solve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
                self,
                mut rhs: Rhs,
                trans: TransposeMode,
            ) -> RlstResult<Rhs> {
                let mut copied = self.into_lapack()?;
                let dim = copied.mat.shape();
                let stride = copied.mat.stride();

                let m = dim.0 as i32;
                let n = dim.1 as i32;
                let lda = stride.1 as i32;
                let ldb = rhs.stride().1;
                let nrhs = rhs.shape().1;

                let info = unsafe {
                    lapacke::$lapack_qr_solve(
                        lapacke::Layout::ColumnMajor,
                        trans as u8,
                        m,
                        n,
                        nrhs as i32,
                        copied.mat.data_mut(),
                        lda,
                        rhs.data_mut(),
                        ldb as i32,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                } else {
                    Ok(rhs)
                }
            }
        }

        impl<
                Mat: RawAccessMut<T = $scalar>
                    + Shape
                    + Stride
                    + std::ops::Index<[usize; 2], Output = $scalar>
                    + std::ops::IndexMut<[usize; 2], Output = $scalar>,
            > QRTrait for QRDecompLapack<$scalar, Mat>
        {
            type T = $scalar;

            fn data(&self) -> &[Self::T] {
                self.data.mat.data()
            }

            fn shape(&self) -> (usize, usize) {
                self.data.mat.shape()
            }

            fn stride(&self) -> (usize, usize) {
                self.data.mat.stride()
            }

            /// Returns Q*RHS
            fn q_mult<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
                &self,
                mut rhs: Rhs,
                trans: TransposeMode,
            ) -> RlstResult<Rhs> {
                if !check_lapack_stride(rhs.shape(), rhs.stride()) {
                    return Err(RlstError::IncompatibleStride);
                } else {
                    let m = rhs.shape().0 as i32;
                    let n = rhs.shape().1 as i32;
                    let lda = self.data.mat.stride().1 as i32;
                    let ldc = rhs.stride().1 as i32;

                    let info = unsafe {
                        lapacke::$lapack_qmut(
                            lapacke::Layout::ColumnMajor,
                            SideMode::Left as u8,
                            trans as u8,
                            m,
                            n,
                            self.data.mat.shape().1 as i32,
                            self.data(),
                            lda,
                            &self.tau,
                            rhs.data_mut(),
                            ldc,
                        )
                    };
                    if info != 0 {
                        return Err(RlstError::LapackError(info));
                    }
                    Ok(rhs)
                }
            }

            fn get_q(&self) -> RlstResult<MatrixD<Self::T>> {
                let mut mat = rlst_mat!(Self::T, (self.shape().0, self.shape().0));

                for index in 0..self.shape().0 {
                    mat[[index, index]] = <Self::T as One>::one();
                }
                self.q_mult(mat, TransposeMode::NoTrans)
            }

            fn get_r(&self) -> RlstResult<MatrixD<Self::T>> {
                let shape = self.shape();
                let dim = std::cmp::min(shape.0, shape.1);
                let mut mat = rlst_mat!(Self::T, (shape.0, shape.1));

                for row in 0..dim {
                    for col in row..shape.1 {
                        mat[[row, col]] = self.data.mat[[row, col]];
                    }
                }
                Ok(mat)
            }

            fn solve_qr<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
                &self,
                mut rhs: Rhs,
                trans: TransposeMode,
            ) -> RlstResult<Rhs> {
                rhs = self.q_mult(rhs, TransposeMode::$trans)?;
                self.get_r()?.linalg().trisolve(
                    rhs,
                    TriangularType::Upper,
                    TriangularDiagonal::NonUnit,
                    trans,
                )
            }
        }
    };
}

qr_decomp_impl!(f32, sgeqrf, sgels, sormqr, Trans);
qr_decomp_impl!(f64, dgeqrf, dgels, dormqr, Trans);
qr_decomp_impl!(c32, cgeqrf, cgels, cunmqr, ConjugateTrans);
qr_decomp_impl!(c64, zgeqrf, zgels, zunmqr, ConjugateTrans);

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use std::ops::Index;

    use crate::linalg::LinAlg;

    use super::*;
    use approx::{assert_abs_diff_eq, AbsDiffEq};
    use rlst_dense::{rlst_col_vec, rlst_mat, Dot};

    #[macro_export]
    macro_rules! assert_approx_matrices {
        ($expected_matrix:expr, $actual_matrix:expr, $epsilon:expr) => {{
            assert_eq!($expected_matrix.shape(), $actual_matrix.shape());
            for row in 0..$expected_matrix.shape().0 {
                for col in 0..$expected_matrix.shape().1 {
                    assert_abs_diff_eq!(
                        $actual_matrix[[row, col]],
                        $expected_matrix[[row, col]],
                        epsilon = $epsilon
                    );
                }
            }
        }};
    }

    macro_rules! test_qr_solve {
        ($scalar:ty, $solver:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let matrix_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let exp_sol = rlst_dense::rlst_rand_col_vec![$scalar, $n];
                let rhs = matrix_a.dot(&exp_sol);
                let rhs = matrix_a
                    .linalg()
                    .$solver(rhs, TransposeMode::NoTrans)
                    .unwrap();


                let mut actual_sol = rlst_col_vec!($scalar, $n);
                actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, $n));

                assert_approx_matrices!(
                    &exp_sol,
                    &actual_sol,
                    1000. * <$scalar as AbsDiffEq>::default_epsilon()
                );
            }
        };
    }

    macro_rules! test_q_unitary {
        ($scalar:ty, $qr:ident, $name:ident, $m:literal, $n:literal, $trans:ident) => {
            #[test]
            fn $name() {
                let rlst_mat = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let qr = rlst_mat.linalg().$qr().unwrap();

                let mut expected_i = rlst_mat!($scalar, ($m, $m));
                for i in 0..$m {
                    expected_i[[i, i]] = <$scalar as One>::one();
                }

                let matrix_q = qr.get_q().unwrap();

                let actual_i_t = qr.q_mult(matrix_q, TransposeMode::$trans).unwrap();
                assert_approx_matrices!(
                    &expected_i,
                    &actual_i_t,
                    1000. * <$scalar as AbsDiffEq>::default_epsilon()
                );
            }
        };
    }

    macro_rules! test_qr_is_a {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let expected_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut rlst_mat = rlst_dense::rlst_mat![$scalar, expected_a.shape()];
                rlst_mat.data_mut().copy_from_slice(expected_a.data());

                let qr = rlst_mat.linalg().$qr_decomp().unwrap();
                let matrix_r = qr.get_r().unwrap();
                let matrix_q = qr.get_q().unwrap();
                let actual_a = matrix_q.dot(&matrix_r);

                assert_approx_matrices!(
                    expected_a,
                    actual_a,
                    1000. * <$scalar as AbsDiffEq>::default_epsilon()
                );
            }
        };
    }

    macro_rules! test_q_mult_r_is_a {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let expected_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let mut rlst_mat = rlst_dense::rlst_mat![$scalar, expected_a.shape()];
                rlst_mat.data_mut().copy_from_slice(expected_a.data());

                let qr = rlst_mat.linalg().$qr_decomp().unwrap();
                let matrix_r = qr.get_r().unwrap();
                let actual_a = qr.q_mult(matrix_r, TransposeMode::NoTrans).unwrap();

                assert_approx_matrices!(
                    &expected_a,
                    &actual_a,
                    1000. * <$scalar as AbsDiffEq>::default_epsilon()
                );
            }
        };
    }

    macro_rules! test_qr_decomp_and_solve {
        ($scalar:ty, $qr_decomp:ident, $name:ident, $m:literal, $n:literal) => {
            #[test]
            fn $name() {
                let matrix_a = rlst_dense::rlst_rand_mat![$scalar, ($m, $n)];
                let exp_sol = rlst_dense::rlst_rand_col_vec![$scalar, $n];
                let rhs = matrix_a.dot(&exp_sol);

                let rhs = matrix_a
                    .linalg()
                    .$qr_decomp()
                    .unwrap()
                    .solve_qr(rhs, TransposeMode::NoTrans)
                    .unwrap();

                let mut actual_sol = rlst_col_vec!($scalar, $n);
                actual_sol.data_mut().copy_from_slice(rhs.get_slice(0, $n));

                assert_approx_matrices!(
                    &exp_sol,
                    &actual_sol,
                    1000. * <$scalar as AbsDiffEq>::default_epsilon()
                );
            }
        };
    }
    test_qr_solve!(f64, qr_and_solve, test_solve_ls_qr_f64, 4, 3);
    test_q_unitary!(f64, qr, test_q_unitary_f64, 4, 3, Trans);
    test_qr_is_a!(f64, qr, test_qr_decomp_f64, 4, 3);
    test_q_mult_r_is_a!(f64, qr, test_q_mult_r_is_a_f64, 4, 3);
    test_qr_decomp_and_solve!(f64, qr, test_qr_decomp_and_solve_f64, 4, 3);
    test_qr_solve!(f32, qr_and_solve, test_solve_ls_qr_f32, 4, 3);
    test_q_unitary!(f32, qr, test_q_unitary_f32, 4, 3, Trans);
    test_qr_is_a!(f32, qr, test_qr_decomp_f32, 4, 3);
    test_q_mult_r_is_a!(f32, qr, test_q_mult_r_is_a_f32, 4, 3);
    test_qr_decomp_and_solve!(f32, qr, test_qr_decomp_and_solve_f32, 4, 3);
    test_qr_solve!(c32, qr_and_solve, test_solve_ls_qr_c32, 4, 3);
    test_q_unitary!(c32, qr, test_q_unitary_c32, 4, 3, ConjugateTrans);
    test_qr_is_a!(c32, qr, test_qr_decomp_c32, 4, 3);
    test_q_mult_r_is_a!(c32, qr, test_q_mult_r_is_a_c32, 4, 3);
    test_qr_decomp_and_solve!(c32, qr, test_qr_decomp_and_solve_c32, 4, 3);
    test_qr_solve!(c64, qr_and_solve, test_solve_ls_qr_c64, 4, 3);
    test_q_unitary!(c64, qr, test_q_unitary_c64, 4, 3, ConjugateTrans);
    test_qr_is_a!(c64, qr, test_qr_decomp_c64, 4, 3);
    test_q_mult_r_is_a!(c64, qr, test_q_mult_r_is_a_c64, 4, 3);
    test_qr_decomp_and_solve!(c64, qr, test_qr_decomp_and_solve_c64, 4, 3);
}
