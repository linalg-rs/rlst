use super::types::*;
use rlst_common::traits::RandomAccessByValue;
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::Shape;

pub trait TriangularSolve {
    type T: Scalar;
    type Out;

    fn triangular_solve<Rhs: RandomAccessByValue<Item = Self::T> + Shape>(
        self,
        rhs: &Rhs,
        tritype: TriangularType,
        tridiag: TriangularDiagonal,
        trans: TransposeMode,
    ) -> RlstResult<Self::Out>;
}
