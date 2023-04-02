//! Trait for LU Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{DataContainerMut, GenericBaseMatrixMut, SizeIdentifier};

pub trait LUDecomp {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn dim(&self) -> (IndexType, IndexType);

    fn solve<Data: DataContainerMut<Item = Self::T>, RS: SizeIdentifier, CS: SizeIdentifier>(
        &self,
        rhs: &mut GenericBaseMatrixMut<Self::T, Data, RS, CS>,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}
