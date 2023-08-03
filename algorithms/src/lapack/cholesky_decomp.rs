//! Interface to Lapack Cholesky Decomposition.
//!
//! The Lapack Cholesky interface uses the Lapack routines `_potrf` to compute a Cholesky decomposition and
//! `_ppsv` to solve a linear system of equations.

use crate::lapack::LapackDataOwned;
use crate::traits::cholesky_decomp::{Cholesky, CholeskyDecomp};
use crate::traits::triangular_solve::TriangularSolve;
use crate::traits::types::TriangularType;
use lapacke;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::MatrixD;

use crate::linalg::{DenseMatrixLinAlgBuilder, LinAlg};
use crate::traits::types::*;
use std::marker::PhantomData;

pub struct CholeskyDecompLapack<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
    data: LapackDataOwned<<Mat as RawAccess>::T, Mat>,
    triangular_type: TriangularType,
    _marker: PhantomData<T>,
}

macro_rules! cholesky_decomp_impl {
    ($scalar:ty, $lapack_potrf:ident, $lapack_ppsv:ident) => {
        impl<'a, Mat: Copy> Cholesky for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar>
                + Shape
                + Stride
                + std::ops::Index<[usize; 2], Output = $scalar>,
        {
            type T = $scalar;
            type Out = CholeskyDecompLapack<$scalar, <Mat as Copy>::Out>;
            /// Compute the LU decomposition.
            fn cholesky(
                self,
                triangular_type: TriangularType,
            ) -> RlstResult<CholeskyDecompLapack<$scalar, <Mat as Copy>::Out>> {
                let mut copied = self.into_lapack()?;
                let shape = copied.mat.shape();
                let stride = copied.mat.stride();

                let m = shape.0;
                let n = shape.1;
                let lda = stride.1 as i32;

                if m != n {
                    return Err(RlstError::MatrixNotSquare(m, n));
                }

                if m == 0 {
                    return Err(RlstError::MatrixIsEmpty((m, n)));
                }

                let info = unsafe {
                    lapacke::$lapack_potrf(
                        lapacke::Layout::ColumnMajor,
                        triangular_type as u8,
                        m as i32,
                        copied.mat.data_mut(),
                        lda,
                    )
                };
                if info == 0 {
                    return Ok(CholeskyDecompLapack {
                        data: copied,
                        triangular_type,
                        _marker: PhantomData,
                    });
                } else {
                    return Err(RlstError::LapackError(info));
                }
            }
        }
    };
}

cholesky_decomp_impl!(f64, dpotrf, dppsv);
cholesky_decomp_impl!(f32, spotrf, sppsv);
cholesky_decomp_impl!(c32, cpotrf, cppsv);
cholesky_decomp_impl!(c64, zpotrf, zppsv);

impl<
        T: Scalar,
        Mat: RawAccessMut<T = T>
            + Shape
            + Stride
            + std::ops::Index<[usize; 2], Output = T>
            + std::ops::IndexMut<[usize; 2], Output = T>
            + RawAccess<T = T>,
    > CholeskyDecomp for CholeskyDecompLapack<T, Mat>
where
    for<'a> DenseMatrixLinAlgBuilder<'a, T, MatrixD<T>>: TriangularSolve<T = T, Out = MatrixD<T>>,
{
    type T = T;
    type Sol = MatrixD<T>;

    fn data(&self) -> &[Self::T] {
        self.data.mat.data()
    }

    fn shape(&self) -> (usize, usize) {
        self.data.mat.shape()
    }

    fn get_l(&self) -> MatrixD<Self::T> {
        let mut result;
        match self.triangular_type {
            TriangularType::Upper => {
                // We need to translate to lower triangular
                let n = self.data.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for row in 0..n {
                    for col in row..n {
                        result[[col, row]] = self.data.mat[[row, col]].conj();
                    }
                }
            }
            TriangularType::Lower => {
                // Just copy over the result
                let n = self.data.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for col in 0..n {
                    for row in col..n {
                        result[[row, col]] = self.data.mat[[row, col]];
                    }
                }
            }
        }
        result
    }

    fn get_u(&self) -> MatrixD<Self::T> {
        let mut result;
        match self.triangular_type {
            TriangularType::Lower => {
                // We need to translate to upper triangular
                let n = self.data.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for col in 0..n {
                    for row in col..n {
                        result[[col, row]] = self.data.mat[[row, col]].conj();
                    }
                }
            }
            TriangularType::Upper => {
                // Just copy over the result
                let n = self.data.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for row in 0..n {
                    for col in row..n {
                        result[[row, col]] = self.data.mat[[row, col]];
                    }
                }
            }
        }
        result
    }

    fn solve<Rhs: RandomAccessByValue<Item = T> + Shape>(
        &self,
        rhs: &Rhs,
    ) -> RlstResult<Self::Sol> {
        let sol = match self.triangular_type {
            TriangularType::Upper => {
                let factor = self.get_u();
                let tmp = factor.linalg().triangular_solve(
                    rhs,
                    TriangularType::Upper,
                    TriangularDiagonal::NonUnit,
                    TransposeMode::ConjugateTrans,
                )?;
                factor.linalg().triangular_solve(
                    &tmp,
                    TriangularType::Upper,
                    TriangularDiagonal::NonUnit,
                    TransposeMode::NoTrans,
                )
            }
            TriangularType::Lower => {
                let factor = self.get_l();
                let tmp = factor.linalg().triangular_solve(
                    rhs,
                    TriangularType::Lower,
                    TriangularDiagonal::NonUnit,
                    TransposeMode::NoTrans,
                )?;
                factor.linalg().triangular_solve(
                    &tmp,
                    TriangularType::Lower,
                    TriangularDiagonal::NonUnit,
                    TransposeMode::ConjugateTrans,
                )
            }
        };
        sol
    }
}

#[cfg(test)]
mod test {
    use crate::linalg::LinAlg;

    use super::*;
    use rlst_dense::Dot;

    #[test]
    fn test_cholesky_f64() {
        let mut rlst_mat = rlst_dense::rlst_mat![f64, (2, 2)];

        rlst_mat[[0, 0]] = 1.0;
        rlst_mat[[0, 1]] = 0.5;
        rlst_mat[[1, 0]] = 0.5;
        rlst_mat[[1, 1]] = 2.0;

        let factor = rlst_mat
            .linalg()
            .cholesky(TriangularType::Lower)
            .unwrap()
            .get_u();

        let actual = factor.view().conj().transpose().eval().dot(&factor);
        rlst_common::assert_matrix_relative_eq!(rlst_mat, actual, 1E-12);
    }

    #[test]
    fn test_cholesky_c32() {
        let mut rlst_mat = rlst_dense::rlst_mat![c32, (2, 2)];

        rlst_mat[[0, 0]] = c32::new(1.0, 0.0);
        rlst_mat[[0, 1]] = c32::new(0.5, 0.1);
        rlst_mat[[1, 0]] = c32::new(0.5, -0.1);
        rlst_mat[[1, 1]] = c32::new(2.0, 0.0);

        let factor = rlst_mat
            .linalg()
            .cholesky(TriangularType::Lower)
            .unwrap()
            .get_u();

        let actual = factor.view().conj().transpose().eval().dot(&factor);
        rlst_common::assert_matrix_relative_eq!(rlst_mat, actual, 1E-5);
    }

    #[test]
    fn test_cholesky_solve_f64() {
        let mut rlst_mat = rlst_dense::rlst_mat![f64, (2, 2)];
        let mut rlst_vec = rlst_dense::rlst_col_vec![f64, 2];

        rlst_mat[[0, 0]] = 1.0;
        rlst_mat[[0, 1]] = 0.5;
        rlst_mat[[1, 0]] = 0.5;
        rlst_mat[[1, 1]] = 2.0;

        rlst_vec[[0, 0]] = 2.3;
        rlst_vec[[1, 0]] = 7.1;

        let rhs = rlst_mat.dot(&rlst_vec);

        let sol = rlst_mat
            .linalg()
            .cholesky(TriangularType::Upper)
            .unwrap()
            .solve(&rhs)
            .unwrap();

        rlst_common::assert_matrix_relative_eq!(rlst_vec, sol, 1E-12);
    }

    #[test]
    fn test_cholesky_solve_c32() {
        let mut rlst_mat = rlst_dense::rlst_mat![c32, (2, 2)];
        let mut rlst_vec = rlst_dense::rlst_col_vec![c32, 2];

        rlst_mat[[0, 0]] = c32::new(1.0, 0.0);
        rlst_mat[[0, 1]] = c32::new(0.5, 0.1);
        rlst_mat[[1, 0]] = c32::new(0.5, -0.1);
        rlst_mat[[1, 1]] = c32::new(2.0, 0.0);

        rlst_vec[[0, 0]] = c32::new(2.3, 0.2);
        rlst_vec[[1, 0]] = c32::new(7.1, 0.9);

        let rhs = rlst_mat.dot(&rlst_vec);

        let sol = rlst_mat
            .linalg()
            .cholesky(TriangularType::Upper)
            .unwrap()
            .solve(&rhs)
            .unwrap();

        rlst_common::assert_matrix_relative_eq!(rlst_vec, sol, 1E-5);
    }
}
