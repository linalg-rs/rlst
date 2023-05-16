use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::{RawAccessMut, Shape, Stride};

use crate::lapack::{TransposeMode, TriangularDiagonal, TriangularType};

pub trait Trisolve {
    type T: Scalar;

    fn trisolve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        self,
        rhs: Rhs,
        tritype: TriangularType,
        tridiag: TriangularDiagonal,
        trans: TransposeMode,
    ) -> RlstResult<Rhs>;
}
