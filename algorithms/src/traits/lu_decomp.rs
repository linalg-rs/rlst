//! Trait for LU Decomposition
use crate::adapter::dense_matrix::{DenseContainer, DenseContainerInterfaceMut};
pub use rlst_common::types::{IndexType, Scalar};

pub trait LUDecomp {
    type T: Scalar;

    fn into(self) -> Self;

    fn data(&self) -> &[Self::T];

    fn dim(&self) -> (IndexType, IndexType);

    fn solve<ContainerImpl: DenseContainerInterfaceMut>(
        &self,
        rhs: &mut DenseContainer<ContainerImpl>,
    );
}
