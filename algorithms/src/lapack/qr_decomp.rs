use crate::traits::qr_decomp::QRDecomposableTrait;
use crate::traits::qr_decomp::QRTrait;
use lapacke::{cunmqr, dormqr, sormqr, zunmqr};
use num::Zero;
#[allow(unused_imports)]
use rlst_common::traits::{Copy, Identity, PermuteColumns};
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_dynamic_mat, MatrixD};
use rlst_dense::{RandomAccessByValue, RawAccess, RawAccessMut, Shape, Stride};

use super::DenseMatrixLinAlgBuilder;
use crate::traits::types::*;

pub struct QRDecompLapack<T: Scalar> {
    mat: MatrixD<T>,
    tau: Vec<T>,
    lda: i32,
    shape: (usize, usize),
    jpvt: Vec<usize>,
}

macro_rules! qr_decomp_impl {
    ($scalar:ty, $lapack_qr:ident, $lapack_qr_pivoting:ident, $lapack_qr_solve:ident, $lapack_qmult:ident) => {
        impl QRDecomposableTrait for DenseMatrixLinAlgBuilder<$scalar> {
            type T = $scalar;
            type Out = QRDecompLapack<$scalar>;
            /// Returns the QR decomposition of the input matrix assuming full rank and using LAPACK xGEQRF
            fn qr(self, pivoting: PivotMode) -> RlstResult<QRDecompLapack<$scalar>> {
                let mut mat = self.mat;
                let shape = mat.shape();
                let stride = mat.stride();

                let m = shape.0 as i32;
                let n = shape.1 as i32;
                let lda = stride.1 as i32;
                let mut tau: Vec<$scalar> =
                    vec![<$scalar as Zero>::zero(); std::cmp::min(shape.0, shape.1)];

                let mut jpvt = vec![0 as i32; shape.1];

                let info = match pivoting {
                    PivotMode::NoPivoting => {
                        // Fill the pivoting array with the identity permutation.
                        // Using 1-based indexing here. Lapack returns 1-based
                        // indexing and fix this when converting to usize array.
                        for (index, item) in jpvt.iter_mut().enumerate() {
                            *item = 1 + index as i32;
                        }
                        unsafe {
                            lapacke::$lapack_qr(
                                lapacke::Layout::ColumnMajor,
                                m,
                                n,
                                mat.data_mut(),
                                lda,
                                &mut tau,
                            )
                        }
                    }
                    PivotMode::WithPivoting => unsafe {
                        lapacke::$lapack_qr_pivoting(
                            lapacke::Layout::ColumnMajor,
                            m,
                            n,
                            mat.data_mut(),
                            lda,
                            &mut jpvt,
                            &mut tau,
                        )
                    },
                };

                if info == 0 {
                    return Ok(QRDecompLapack {
                        mat,
                        tau,
                        lda,
                        shape,
                        jpvt: jpvt.iter().map(|&item| item as usize - 1).collect(),
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

                let mut mat = self.mat;
                let shape = mat.shape();

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

                let stride = mat.stride();

                let ldb = std::cmp::max(shape.0, shape.1);
                let m = shape.0 as i32;
                let n = shape.1 as i32;
                let lda = stride.1 as i32;

                let nrhs = rhs.shape().1;

                // Create the rhs with the right dimension for any case.
                let mut work_rhs = rlst_dynamic_mat![Self::T, (ldb, nrhs)];

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
                        mat.data_mut(),
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

                    let mut sol = rlst_dynamic_mat![Self::T, (x_rows, nrhs)];

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

        impl QRTrait for QRDecompLapack<$scalar> {
            type T = $scalar;
            type Q = MatrixD<Self::T>;
            type R = MatrixD<Self::T>;

            fn q(&self, mode: QrMode) -> RlstResult<MatrixD<Self::T>> {
                let q_dim = if self.shape.0 < self.shape.1 {
                    // Fewer rows than columns
                    (self.shape.0, self.shape.0)
                } else {
                    // rows >= columns
                    match mode {
                        QrMode::Reduced => {
                            (self.shape.0, std::cmp::min(self.shape.0, self.shape.1))
                        }
                        QrMode::Full => (self.shape.0, self.shape.0),
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
                        self.mat.data(),
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
                let shape = self.mat.shape();
                let dim = std::cmp::min(shape.0, shape.1);
                let mut mat = rlst_dynamic_mat!(Self::T, (dim, shape.1));

                for row in 0..dim {
                    for col in row..shape.1 {
                        mat[[row, col]] = self.mat[[row, col]];
                    }
                }
                Ok(mat)
            }

            fn permutation(&self) -> &Vec<usize> {
                &self.jpvt
            }
        }
    };
}

qr_decomp_impl!(f32, sgeqrf, sgeqp3, sgels, sormqr);
qr_decomp_impl!(f64, dgeqrf, dgeqp3, dgels, dormqr);
qr_decomp_impl!(c32, cgeqrf, cgeqp3, cgels, cunmqr);
qr_decomp_impl!(c64, zgeqrf, zgeqp3, zgels, zunmqr);

#[cfg(test)]
mod test {
    #![allow(unused_imports)]
    use std::ops::Index;

    use crate::linalg::LinAlg;
    use crate::traits::lu_decomp::LU;
    use crate::traits::norm2::Norm2;
    use paste::paste;
    use rlst_common::assert_matrix_abs_diff_eq;
    use rlst_common::assert_matrix_relative_eq;
    use rlst_common::traits::*;

    use super::*;
    use crate::traits::lu_decomp::LUDecomp;
    use rlst_dense::{rlst_col_vec, rlst_dynamic_mat, rlst_rand_col_vec, rlst_rand_mat, Dot};

    macro_rules! qr_tests {
        ($ScalarType:ident, $PivotMode:expr, $trans:expr, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_thick_qr_ $ScalarType _ $PivotMode>]() {
                    // QR Decomposition of a thick matrix.

                    let mut mat = rlst_dynamic_mat![$ScalarType, (3, 5)];
                    mat.fill_from_seed_equally_distributed(0);

                    let pivot_mode;
                    if $PivotMode == "nopivot" {
                        pivot_mode = PivotMode::NoPivoting;
                    } else {
                        pivot_mode = PivotMode::WithPivoting;
                    }

                    let qr = mat.linalg().qr(pivot_mode).unwrap();

                    let q = qr.q(QrMode::Reduced).unwrap();
                    let r = qr.r().unwrap();
                    let jpvt = qr.permutation().iter().map(|&item| item as usize).collect::<Vec<usize>>();

                    // Test dimensions

                    assert_eq!(q.shape(), (3, 3));
                    assert_eq!(r.shape(), (3, 5));

                    // Test orthogonality of Q

                    let actual = q.view().conj().transpose().eval().dot(&q).eval();
                    let expected = MatrixD::<$ScalarType>::identity((3, 3));

                    assert_eq!(actual.shape(), (3, 3));
                    assert_matrix_abs_diff_eq!(actual, expected, $tol);

                    // Test that r is triangular

                    for row_index in 0..r.shape().0 {
                        for col_index in 0..row_index {
                            assert_eq!(r[[row_index, col_index]], <$ScalarType>::zero());
                        }
                    }

                    // Test backward error

                    let actual = q.dot(&r);
                    assert_matrix_relative_eq!(actual, mat.permute_columns(&jpvt), $tol);
                }

                #[test]
                fn [<test_thin_qr_ $ScalarType _ $PivotMode>]() {
                    // QR Decomposition of a thin matrix.

                    let mut mat = rlst_dynamic_mat![$ScalarType, (5, 3)];
                    mat.fill_from_seed_equally_distributed(0);

                    let pivot_mode;
                    if $PivotMode == "nopivot" {
                        pivot_mode = PivotMode::NoPivoting;
                    } else {
                        pivot_mode = PivotMode::WithPivoting;
                    }

                    let qr = mat.linalg().qr(pivot_mode).unwrap();

                    let q_thin = qr.q(QrMode::Reduced).unwrap();
                    let q_thick = qr.q(QrMode::Full).unwrap();
                    let r = qr.r().unwrap();
                    let jpvt = qr.permutation().iter().map(|&item| item as usize).collect::<Vec<usize>>();


                    // Test dimensions

                    assert_eq!(q_thin.shape(), (5, 3));
                    assert_eq!(q_thick.shape(), (5, 5));
                    assert_eq!(r.shape(), (3, 3));

                    // Test orthogonality of Q

                    let actual_thin = q_thin.view().conj().transpose().eval().dot(&q_thin);
                    let expected_thin = MatrixD::<$ScalarType>::identity((3, 3));

                    let actual_thick = q_thick.view().conj().transpose().eval().dot(&q_thick);
                    let expected_thick = MatrixD::<$ScalarType>::identity((5, 5));

                    assert_eq!(actual_thin.shape(), (3, 3));
                    assert_eq!(actual_thick.shape(), (5, 5));

                    assert_matrix_abs_diff_eq!(actual_thin, expected_thin, $tol);
                    assert_matrix_abs_diff_eq!(actual_thick, expected_thick, $tol);

                    // Test that r is triangular

                    for row_index in 0..r.shape().0 {
                        for col_index in 0..row_index {
                            assert_eq!(r[[row_index, col_index]], <$ScalarType>::zero());
                        }
                    }

                    // Test backward error

                    let actual = q_thin.dot(&r);
                    assert_matrix_relative_eq!(actual, mat.permute_columns(&jpvt), $tol);
                }

                #[test]
                fn [<test_least_squares_solve_thin_ $ScalarType _ $PivotMode>]() {
                    // Test notrans

                    let mut mat = rlst_dynamic_mat![$ScalarType, (5, 3)];
                    mat.fill_from_seed_equally_distributed(0);

                    let mut rhs = rlst_dynamic_mat![$ScalarType, (5, 2)];
                    rhs.fill_from_seed_equally_distributed(2);

                    let normal_lhs = mat.view().conj().transpose().eval().dot(&mat);
                    let normal_rhs = mat.view().conj().transpose().eval().dot(&rhs);

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

                    assert_matrix_relative_eq!(expected, actual, $tol);

                    let mut rhs = rlst_col_vec![$ScalarType, 3];
                    rhs.fill_from_seed_equally_distributed(2);
                    let sol = mat.linalg().solve_least_squares(&rhs, $trans).unwrap();

                    let res_norm = (mat.conj().transpose().eval().dot(&sol) - &rhs)
                        .linalg()
                        .norm2()
                        .unwrap();

                    assert!(res_norm / rhs.linalg().norm2().unwrap() < $tol);
                }

                #[test]
                fn [<test_least_squares_solve_thick_no_conj_trans_ $ScalarType _ $PivotMode>]() {
                    let mut mat = rlst_dynamic_mat![$ScalarType, (3, 5)];
                    mat.fill_from_seed_equally_distributed(0);

                    let mut rhs = rlst_col_vec![$ScalarType, 3];
                    rhs.fill_from_seed_equally_distributed(2);

                    let sol = mat
                        .linalg()
                        .solve_least_squares(&rhs, TransposeMode::NoTrans)
                        .unwrap();

                    let res_norm = (&rhs - &mat.dot(&sol)).linalg().norm2().unwrap();

                    assert!(res_norm / rhs.linalg().norm2().unwrap() < $tol);

                    // Test transpose mode

                    let mut rhs = rlst_col_vec![$ScalarType, 5];
                    rhs.fill_from_seed_equally_distributed(2);

                    let normal_lhs = mat.dot(&mat.view().conj().transpose().eval());
                    let normal_rhs = mat.dot(&rhs);

                    let expected = normal_lhs
                        .linalg()
                        .lu()
                        .unwrap()
                        .solve(&normal_rhs, TransposeMode::NoTrans)
                        .unwrap();

                    let actual = mat.linalg().solve_least_squares(&rhs, $trans).unwrap();

                    assert_matrix_relative_eq!(expected, actual, $tol);
                }
            }
        };
    }

    qr_tests!(f32, "nopivot", TransposeMode::Trans, 1E-5);
    qr_tests!(f64, "nopivot", TransposeMode::Trans, 1E-13);
    qr_tests!(c32, "nopivot", TransposeMode::ConjugateTrans, 1E-5);
    qr_tests!(c64, "nopivot", TransposeMode::ConjugateTrans, 1E-13);

    qr_tests!(f32, "pivot", TransposeMode::Trans, 1E-5);
    qr_tests!(f64, "pivot", TransposeMode::Trans, 1E-13);
    qr_tests!(c32, "pivot", TransposeMode::ConjugateTrans, 1E-5);
    qr_tests!(c64, "pivot", TransposeMode::ConjugateTrans, 1E-13);
}
