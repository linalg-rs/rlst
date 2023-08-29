use super::types::*;
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::{Matrix, MatrixD, MatrixImplTrait, SizeIdentifier};

pub trait TriangularSolve {
    type T: Scalar;

    fn triangular_solve<S: SizeIdentifier, MatImpl: MatrixImplTrait<Self::T, S>>(
        &self,
        rhs: &Matrix<Self::T, MatImpl, S>,
        tritype: TriangularType,
        tridiag: TriangularDiagonal,
        trans: TransposeMode,
    ) -> RlstResult<MatrixD<Self::T>>;
}
