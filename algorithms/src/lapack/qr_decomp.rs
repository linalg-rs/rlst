use crate::linalg::LinAlg;
use crate::traits::qr_decomp_trait::{Mode, QRTrait};
use crate::traits::trisolve_trait::Trisolve;
use crate::{lapack::LapackDataOwned, traits::qr_decomp_trait::QRDecomposableTrait};
use lapacke::{cunmqr, dormqr, sormqr, zunmqr};
use num::{One, Zero};
use rlst_common::traits::{Copy, Identity};
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_mat, MatrixD};
use rlst_dense::{RandomAccessByValue, RawAccess, RawAccessMut, Shape, Stride};

use super::{
    check_lapack_stride, DenseMatrixLinAlgBuilder, SideMode, TransposeMode, TriangularDiagonal,
    TriangularType,
};

pub struct QRDecompLapack<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
    data: LapackDataOwned<T, Mat>,
    tau: Vec<T>,
    lda: i32,
    shape: (usize, usize),
    stride: (usize, usize),
}

macro_rules! qr_decomp_impl {
    ($scalar:ty, $lapack_qr:ident, $lapack_qr_solve:ident, $lapack_qmult:ident) => {
        impl<'a, Mat: Copy> QRDecomposableTrait for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar> + RawAccess<T = $scalar> + Shape + Stride,
        {
            type T = $scalar;
            type Out = QRDecompLapack<$scalar, <Mat as Copy>::Out>;
            /// Returns the QR decomposition of the input matrix assuming full rank and using LAPACK xGEQRF
            fn qr(self) -> RlstResult<QRDecompLapack<$scalar, <Mat as Copy>::Out>> {
                let mut copied = self.into_lapack()?;
                let shape = copied.mat.shape();
                let stride = copied.mat.stride();

                let m = shape.0 as i32;
                let n = shape.1 as i32;
                let lda = stride.1 as i32;
                let mut tau: Vec<$scalar> =
                    vec![<$scalar as Zero>::zero(); std::cmp::min(shape.0, shape.1)];

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
                    return Ok(QRDecompLapack {
                        data: copied,
                        tau,
                        lda,
                        shape,
                        stride,
                    });
                } else {
                    return Err(RlstError::LapackError(info));
                }
            }

            fn solve_least_squares<Rhs: RandomAccessByValue<Item = Self::T> + Shape + Stride>(
                self,
                rhs: &Rhs,
                trans: TransposeMode,
            ) -> RlstResult<MatrixD<Self::T>> {
                if rhs.is_empty() {
                    return Err(RlstError::MatrixIsEmpty(rhs.shape()));
                }

                let mut copied = self.into_lapack()?;
                let shape = copied.mat.shape();

                let expected_rhs_rows = match trans {
                    TransposeMode::NoTrans => shape.0,
                    _ => shape.1,
                };

                if rhs.shape().0 != expected_rhs_rows {
                    return Err(RlstError::GeneralError(format!(
                        "rhs has wrong dimension. Expected {}. Actual {}",
                        expected_rhs_rows,
                        rhs.shape().0
                    )));
                }

                let stride = copied.mat.stride();

                let ldb = std::cmp::max(shape.0, shape.1);
                let m = shape.0 as i32;
                let n = shape.1 as i32;
                let lda = stride.1 as i32;

                let nrhs = rhs.shape().1;

                // Create the rhs with the right dimension for any case.
                let mut work_rhs = rlst_mat![Self::T, (ldb, nrhs)];

                // Copy rhs into work_rhs
                for col_index in 0..nrhs {
                    for row_index in 0..rhs.shape().0 {
                        work_rhs[[row_index, col_index]] =
                            rhs.get_value(row_index, col_index).unwrap();
                    }
                }

                let info = unsafe {
                    lapacke::$lapack_qr_solve(
                        lapacke::Layout::ColumnMajor,
                        trans as u8,
                        m,
                        n,
                        nrhs as i32,
                        copied.mat.data_mut(),
                        lda,
                        work_rhs.data_mut(),
                        ldb as i32,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                } else {
                    // Create the output array

                    let x_rows = match trans {
                        TransposeMode::NoTrans => n,
                        TransposeMode::Trans => m,
                        TransposeMode::ConjugateTrans => m,
                    } as usize;

                    let mut sol = rlst_mat![Self::T, (x_rows, nrhs)];

                    // Copy solution back

                    for col_index in 0..nrhs {
                        for row_index in 0..x_rows {
                            sol[[row_index, col_index]] = work_rhs[[row_index, col_index]];
                        }
                    }

                    Ok(sol)
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
            type Q = MatrixD<Self::T>;
            type R = MatrixD<Self::T>;

            fn q(&self, mode: Mode) -> RlstResult<MatrixD<Self::T>> {
                let q_dim = if self.shape.0 < self.shape.1 {
                    // Fewer rows than columns
                    (self.shape.0, self.shape.0)
                } else {
                    // rows >= columns
                    match mode {
                        Mode::Reduced => (self.shape.0, std::cmp::min(self.shape.0, self.shape.1)),
                        Mode::Full => (self.shape.0, self.shape.0),
                    }
                };

                let mut q = MatrixD::<Self::T>::identity(q_dim);

                let info = unsafe {
                    $lapack_qmult(
                        lapacke::Layout::ColumnMajor,
                        SideMode::Left as u8,
                        TransposeMode::NoTrans as u8,
                        q_dim.0 as i32,
                        q_dim.1 as i32,
                        self.tau.len() as i32,
                        self.data.mat.data(),
                        self.lda,
                        self.tau.as_slice(),
                        q.data_mut(),
                        q_dim.0 as i32,
                    )
                };

                if info == 0 {
                    Ok(q)
                } else {
                    Err(RlstError::LapackError(info))
                }
            }

            fn r(&self) -> RlstResult<MatrixD<Self::T>> {
                let shape = self.data.mat.shape();
                let dim = std::cmp::min(shape.0, shape.1);
                let mut mat = rlst_mat!(Self::T, (dim, shape.1));

                for row in 0..dim {
                    for col in row..shape.1 {
                        mat[[row, col]] = self.data.mat[[row, col]];
                    }
                }
                Ok(mat)
            }
        }
    };
}

qr_decomp_impl!(f32, sgeqrf, sgels, sormqr);
qr_decomp_impl!(f64, dgeqrf, dgels, dormqr);
qr_decomp_impl!(c32, cgeqrf, cgels, cunmqr);
qr_decomp_impl!(c64, zgeqrf, zgels, zunmqr);

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use std::ops::Index;

    use crate::linalg::LinAlg;
    use crate::traits::lu_decomp::LU;
    use crate::traits::norm2::Norm2;
    use rlst_common::assert_matrix_abs_diff_eq;
    use rlst_common::assert_matrix_relative_eq;
    use rlst_common::traits::ConjTranspose;
    use rlst_common::traits::Eval;

    use super::*;
    use crate::traits::lu_decomp::LUDecomp;
    use rlst_dense::{rlst_col_vec, rlst_mat, rlst_rand_col_vec, rlst_rand_mat, Dot};

    #[test]
    fn test_thick_qr() {
        // QR Decomposition of a thick matrix.

        let mat = rlst_rand_mat![c64, (3, 5)];

        let qr = mat.linalg().qr().unwrap();

        let q = qr.q(Mode::Reduced).unwrap();
        let r = qr.r().unwrap();

        // Test dimensions

        assert_eq!(q.shape(), (3, 3));
        assert_eq!(r.shape(), (3, 5));

        // Test orthogonality of Q

        let actual = q.conj_transpose().dot(&q);
        let expected = MatrixD::<c64>::identity((3, 3));

        assert_eq!(actual.shape(), (3, 3));
        assert_matrix_abs_diff_eq!(actual, expected, 1E-12);

        // Test that r is triangular

        for row_index in 0..r.shape().0 {
            for col_index in 0..row_index {
                assert_eq!(r[[row_index, col_index]], c64::zero());
            }
        }

        // Test backward error

        let actual = q.dot(&r);
        assert_matrix_relative_eq!(actual, mat.eval(), 1E-14);
    }

    #[test]
    fn test_thin_qr() {
        // QR Decomposition of a thin matrix.

        let mat = rlst_rand_mat![c64, (5, 3)];

        let qr = mat.linalg().qr().unwrap();

        let q_thin = qr.q(Mode::Reduced).unwrap();
        let q_thick = qr.q(Mode::Full).unwrap();
        let r = qr.r().unwrap();

        // Test dimensions

        assert_eq!(q_thin.shape(), (5, 3));
        assert_eq!(q_thick.shape(), (5, 5));
        assert_eq!(r.shape(), (3, 3));

        // Test orthogonality of Q

        let actual_thin = q_thin.conj_transpose().dot(&q_thin);
        let expected_thin = MatrixD::<c64>::identity((3, 3));

        let actual_thick = q_thick.conj_transpose().dot(&q_thick);
        let expected_thick = MatrixD::<c64>::identity((5, 5));

        assert_eq!(actual_thin.shape(), (3, 3));
        assert_eq!(actual_thick.shape(), (5, 5));

        assert_matrix_abs_diff_eq!(actual_thin, expected_thin, 1E-12);
        assert_matrix_abs_diff_eq!(actual_thick, expected_thick, 1E-12);

        // Test that r is triangular

        for row_index in 0..r.shape().0 {
            for col_index in 0..row_index {
                assert_eq!(r[[row_index, col_index]], c64::zero());
            }
        }

        // Test backward error

        let actual = q_thin.dot(&r);
        assert_matrix_relative_eq!(actual, mat.eval(), 1E-14);
    }

    #[test]
    fn test_least_squares_solve_thin() {
        // Test notrans

        let mat = rlst_rand_mat![c64, (5, 3)];

        let rhs = rlst_rand_mat![c64, (5, 2)];

        let normal_lhs = mat.conj_transpose().dot(&mat);
        let normal_rhs = mat.conj_transpose().dot(&rhs);

        let expected = normal_lhs
            .linalg()
            .lu()
            .unwrap()
            .solve(&normal_rhs, TransposeMode::NoTrans)
            .unwrap();

        let actual = mat
            .linalg()
            .solve_least_squares(&rhs, TransposeMode::NoTrans)
            .unwrap();

        assert_matrix_relative_eq!(expected, actual, 1E-12);

        let rhs = rlst_rand_col_vec![c64, 3];
        let sol = mat
            .linalg()
            .solve_least_squares(&rhs, TransposeMode::ConjugateTrans)
            .unwrap();

        let res_norm = (mat.conj_transpose().dot(&sol) - &rhs)
            .linalg()
            .norm2()
            .unwrap();

        assert!(res_norm / rhs.linalg().norm2().unwrap() < 1E-12);
    }

    #[test]
    fn test_least_squares_solve_thick_no_conj_trans() {
        let mat = rlst_rand_mat![c64, (3, 5)];

        let rhs = rlst_rand_col_vec![c64, 3];

        let sol = mat
            .linalg()
            .solve_least_squares(&rhs, TransposeMode::NoTrans)
            .unwrap();

        let res_norm = (&rhs - &mat.dot(&sol)).linalg().norm2().unwrap();

        assert!(res_norm / rhs.linalg().norm2().unwrap() < 1E-12);

        // Test transpose mode

        let rhs = rlst_rand_col_vec![c64, 5];

        let normal_lhs = mat.dot(&mat.conj_transpose());
        let normal_rhs = mat.dot(&rhs);

        let expected = normal_lhs
            .linalg()
            .lu()
            .unwrap()
            .solve(&normal_rhs, TransposeMode::NoTrans)
            .unwrap();

        let actual = mat
            .linalg()
            .solve_least_squares(&rhs, TransposeMode::ConjugateTrans)
            .unwrap();

        assert_matrix_relative_eq!(expected, actual, 1E-12);
    }
}
