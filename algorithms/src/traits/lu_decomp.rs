//! Trait for LU Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{DataContainerMut, GenericBaseMatrixMut, SizeIdentifier};

pub trait LUDecomp {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn shape(&self) -> (usize, usize);

    fn solve<Data: DataContainerMut<Item = Self::T>, RhsR: SizeIdentifier, RhsC: SizeIdentifier>(
        &self,
        rhs: &mut GenericBaseMatrixMut<Self::T, Data, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}
