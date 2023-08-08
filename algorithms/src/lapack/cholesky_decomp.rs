//! Interface to Lapack Cholesky Decomposition.
//!
//! The Lapack Cholesky interface uses the Lapack routines `_potrf` to compute a Cholesky decomposition and
//! `_ppsv` to solve a linear system of equations.

use crate::traits::cholesky_decomp::{Cholesky, CholeskyDecomp};
use crate::traits::triangular_solve::TriangularSolve;
use crate::traits::types::TriangularType;
use lapacke;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{Matrix, MatrixD, MatrixImplTrait, SizeIdentifier};

use crate::linalg::{DenseMatrixLinAlgBuilder, LinAlg};
use crate::traits::types::*;
use std::marker::PhantomData;

pub struct CholeskyDecompLapack<T: Scalar> {
    mat: MatrixD<T>,
    triangular_type: TriangularType,
    _marker: PhantomData<T>,
}

macro_rules! cholesky_decomp_impl {
    ($scalar:ty, $lapack_potrf:ident, $lapack_ppsv:ident) => {
        impl Cholesky for DenseMatrixLinAlgBuilder<$scalar> {
            type T = $scalar;
            type Out = CholeskyDecompLapack<$scalar>;
            /// Compute the LU decomposition.
            fn cholesky(self) -> RlstResult<CholeskyDecompLapack<$scalar>> {
                let triangular_type = TriangularType::Upper;

                let mut mat = self.mat;
                let shape = mat.shape();
                let stride = mat.stride();

                let m = shape.0;
                let lda = stride.1 as i32;

                if !mat.is_hermitian() {
                    return Err(RlstError::MatrixNotHermitian);
                }

                let info = unsafe {
                    lapacke::$lapack_potrf(
                        lapacke::Layout::ColumnMajor,
                        triangular_type as u8,
                        m as i32,
                        mat.data_mut(),
                        lda,
                    )
                };
                if info == 0 {
                    return Ok(CholeskyDecompLapack {
                        mat,
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

impl<T: Scalar> CholeskyDecomp for CholeskyDecompLapack<T>
where
    DenseMatrixLinAlgBuilder<T>: TriangularSolve<T = T>,
{
    type T = T;
    type Sol = MatrixD<T>;

    fn data(&self) -> &[Self::T] {
        self.mat.data()
    }

    fn shape(&self) -> (usize, usize) {
        self.mat.shape()
    }

    fn get_l(&self) -> MatrixD<Self::T> {
        let mut result;
        match self.triangular_type {
            TriangularType::Upper => {
                // We need to translate to lower triangular
                let n = self.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for row in 0..n {
                    for col in row..n {
                        result[[col, row]] = self.mat[[row, col]].conj();
                    }
                }
            }
            TriangularType::Lower => {
                // Just copy over the result
                let n = self.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for col in 0..n {
                    for row in col..n {
                        result[[row, col]] = self.mat[[row, col]];
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
                let n = self.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for col in 0..n {
                    for row in col..n {
                        result[[col, row]] = self.mat[[row, col]].conj();
                    }
                }
            }
            TriangularType::Upper => {
                // Just copy over the result
                let n = self.mat.shape().0;
                result = rlst_dense::rlst_mat![Self::T, (n, n)];
                for row in 0..n {
                    for col in row..n {
                        result[[row, col]] = self.mat[[row, col]];
                    }
                }
            }
        }
        result
    }

    fn solve<MatImpl: MatrixImplTrait<Self::T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>(
        &self,
        rhs: &Matrix<Self::T, MatImpl, RS, CS>,
    ) -> RlstResult<Self::Sol>
// where
    //     Matrix<Self::T, MatImpl, RS, CS>: TriangularSolve<T = Self::T>,
    {
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
    use rlst_dense::types::Scalar;
    use rlst_dense::Dot;

    use paste::paste;

    macro_rules! test_impl {
        ($scalar:ty, $tol:expr) => {
            paste! {
                    #[test]
                    fn [<test_cholesky_$scalar>]() {
                        let mut rlst_mat = rlst_dense::rlst_mat![$scalar, (2, 2)];

                        rlst_mat.fill_from_seed_equally_distributed(0);
                        rlst_mat[[1, 0]] = rlst_mat[[0, 1]].conj();
                        rlst_mat[[0, 0]] = <$scalar as Scalar>::from_real(3.0);
                        rlst_mat[[1, 1]] = <$scalar as Scalar>::from_real(5.0);


                        let lower = rlst_mat
                            .linalg()
                            .cholesky()
                            .unwrap()
                            .get_l();

                        let upper = rlst_mat
                            .linalg()
                            .cholesky()
                            .unwrap()
                            .get_u();

                        let actual = lower.dot(&upper);
                        rlst_common::assert_matrix_relative_eq!(rlst_mat, actual, $tol);

                    }

                #[test]
                fn [<test_cholesky_solve_$scalar>]() {
                    let mut rlst_mat = rlst_dense::rlst_mat![$scalar, (2, 2)];
                    let mut rlst_vec = rlst_dense::rlst_col_vec![$scalar, 2];

                    rlst_mat.fill_from_seed_equally_distributed(0);
                    rlst_mat[[1, 0]] = rlst_mat[[0, 1]].conj();
                    rlst_mat[[0, 0]] = <$scalar as Scalar>::from_real(3.0);
                    rlst_mat[[1, 1]] = <$scalar as Scalar>::from_real(5.0);

                    rlst_vec.fill_from_seed_equally_distributed(1);

                    let rhs = rlst_mat.dot(&rlst_vec);

                    // Test Lower

                    let sol = rlst_mat
                        .linalg()
                        .cholesky()
                        .unwrap()
                        .solve(&rhs)
                        .unwrap();

                    rlst_common::assert_matrix_relative_eq!(rlst_vec, sol, $tol);

                }
            }
        };
    }

    test_impl!(f32, 1E-5);
    test_impl!(f64, 1E-12);
    test_impl!(c32, 1E-5);
    test_impl!(c64, 1E-12);
}
