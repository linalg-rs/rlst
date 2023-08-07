use super::types::*;
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::{Matrix, MatrixD, MatrixImplTrait, SizeIdentifier};

pub trait TriangularSolve {
    type T: Scalar;

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
    ) -> RlstResult<MatrixD<Self::T>>;
}
