//! Trait for LU Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{IndexType, RlstError, RlstResult, Scalar};

pub trait LUDecomp {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn dim(&self) -> (IndexType, IndexType);

    // fn solve<VecImpl: DenseContainerInterfaceMut<T = Self::T>>(
    //     &self,
    //     rhs: &mut DenseContainer<VecImpl>,
    //     trans: TransposeMode,
    // ) -> RlstResult<()>;
}
