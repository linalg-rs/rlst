//! Trait for LU Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{IndexType, RlstError, RlstResult, Scalar};
use rlst_dense::{DataContainerMut, GenericBaseMatrixMut, SizeIdentifier};

pub trait QRDecomp {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn dim(&self) -> (IndexType, IndexType);

    fn solve<Data: DataContainerMut<Item = Self::T>, RhsR: SizeIdentifier, RhsC: SizeIdentifier>(
        &mut self,
        rhs: &mut GenericBaseMatrixMut<Self::T, Data, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}
