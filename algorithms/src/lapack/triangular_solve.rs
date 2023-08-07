use rlst_common::types::{c32, c64, RlstError, RlstResult};
use rlst_dense::MatrixD;
use rlst_dense::{Matrix, MatrixImplTrait, RawAccess, RawAccessMut, Shape, SizeIdentifier, Stride};

use lapacke;

use crate::traits::triangular_solve::TriangularSolve;

use super::DenseMatrixLinAlgBuilder;

use crate::traits::types::{TransposeMode, TriangularDiagonal, TriangularType};

// Need to distinguish real and complex implementation
// as for the real implementation we need to convert TransposeMode::ConjTrans
// into TransposeMode::Trans.

macro_rules! trisolve_real_impl {
    ($scalar:ty, $lapack_trisolve:ident) => {
        impl TriangularSolve for DenseMatrixLinAlgBuilder<$scalar> {
            type T = $scalar;
            fn triangular_solve<
                RS: SizeIdentifier,
                CS: SizeIdentifier,
                MatImpl: MatrixImplTrait<Self::T, RS, CS>,
            >(
                &self,
                rhs: &Matrix<Self::T, MatImpl, RS, CS>,
                tritype: TriangularType,
                tridiag: TriangularDiagonal,
                trans: TransposeMode,
            ) -> RlstResult<MatrixD<Self::T>> {
                if rhs.shape().0 == 0 || rhs.shape().1 == 0 {
                    return Err(RlstError::MatrixIsEmpty(rhs.shape()));
                }

                if rhs.shape().0 != self.mat.shape().1 {
                    return Err(RlstError::GeneralError(format!(
                        "Incompatible rhs. Expected rows {}. Actual rows {}",
                        rhs.shape().0,
                        self.mat.shape().1
                    )));
                }

                let mat = &self.mat;

                //let mat = self.mat.to_dyn_matrix();

                let n = mat.shape().1 as i32;
                let mut sol = MatrixD::<$scalar>::from_other(rhs);
                let nrhs = sol.shape().1 as i32;
                let ldb = sol.stride().1 as i32;
                let lda = n;

                // Ensure that conjugate transpose is accepted also
                // for real matrices.

                let trans = match trans {
                    TransposeMode::NoTrans => TransposeMode::NoTrans,
                    TransposeMode::Trans => TransposeMode::Trans,
                    TransposeMode::ConjugateTrans => TransposeMode::Trans,
                };

                let info = unsafe {
                    lapacke::$lapack_trisolve(
                        lapacke::Layout::ColumnMajor,
                        tritype as u8,
                        trans as u8,
                        tridiag as u8,
                        n,
                        nrhs,
                        mat.data(),
                        lda,
                        sol.data_mut(),
                        ldb,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                } else {
                    Ok(sol)
                }
            }
        }
    };
}

macro_rules! trisolve_complex_impl {
    ($scalar:ty, $lapack_trisolve:ident) => {
        impl TriangularSolve for DenseMatrixLinAlgBuilder<$scalar> {
            type T = $scalar;
            fn triangular_solve<
                RS: SizeIdentifier,
                CS: SizeIdentifier,
                MatImpl: MatrixImplTrait<Self::T, RS, CS>,
            >(
                &self,
                rhs: &Matrix<Self::T, MatImpl, RS, CS>,
                tritype: TriangularType,
                tridiag: TriangularDiagonal,
                trans: TransposeMode,
            ) -> RlstResult<MatrixD<Self::T>> {
                if rhs.shape().0 == 0 || rhs.shape().1 == 0 {
                    return Err(RlstError::MatrixIsEmpty(rhs.shape()));
                }

                if rhs.shape().0 != self.mat.shape().1 {
                    return Err(RlstError::GeneralError(format!(
                        "Incompatible rhs. Expected rows {}. Actual rows {}",
                        rhs.shape().0,
                        self.mat.shape().1
                    )));
                }

                if self.mat.shape().0 != self.mat.shape().1 {
                    return Err(RlstError::MatrixNotSquare(
                        self.mat.shape().0,
                        self.mat.shape().1,
                    ));
                }

                let mat = &self.mat;

                let n = mat.shape().1 as i32;
                let mut sol = MatrixD::<$scalar>::from_other(rhs);
                let nrhs = sol.shape().1 as i32;
                let ldb = sol.stride().1 as i32;
                let lda = n;

                let info = unsafe {
                    lapacke::$lapack_trisolve(
                        lapacke::Layout::ColumnMajor,
                        tritype as u8,
                        trans as u8,
                        tridiag as u8,
                        n,
                        nrhs,
                        mat.data(),
                        lda,
                        sol.data_mut(),
                        ldb,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                } else {
                    Ok(sol)
                }
            }
        }
    };
}

trisolve_real_impl!(f32, strtrs);
trisolve_real_impl!(f64, dtrtrs);
trisolve_complex_impl!(c32, ctrtrs);
trisolve_complex_impl!(c64, ztrtrs);

#[allow(unused_imports)]
#[cfg(test)]
mod test {
    use num::Zero;
    use rlst_dense::{rlst_col_vec, rlst_mat, Dot};

    use super::*;
    use crate::linalg::LinAlg;
    use rlst_common::assert_matrix_relative_eq;

    macro_rules! test_trisolve {
        ($scalar:ty, $name:ident, $tol:expr) => {
            #[test]
            fn $name() {
                let mut mat_a = rlst_mat![$scalar, (4, 4)];
                mat_a.fill_from_seed_equally_distributed(0);
                for row in 0..mat_a.shape().0 {
                    for col in 0..row {
                        mat_a[[row, col]] = <$scalar as Zero>::zero();
                    }
                }
                let mut exp_sol = rlst_col_vec![$scalar, 4];
                exp_sol.fill_from_seed_equally_distributed(1);
                let rhs = mat_a.dot(&exp_sol);
                let sol = mat_a
                    .linalg()
                    .triangular_solve(
                        &rhs,
                        TriangularType::Upper,
                        TriangularDiagonal::NonUnit,
                        TransposeMode::NoTrans,
                    )
                    .unwrap();

                assert_matrix_relative_eq!(exp_sol, sol, $tol);
            }
        };
    }

    test_trisolve!(f32, test_trisolve_f32, 1E-6);
    test_trisolve!(f64, test_trisolve_f64, 1E-12);
    test_trisolve!(c32, test_trisolve_c32, 1E-5);
    test_trisolve!(c64, test_trisolve_c64, 1E-12);
}
