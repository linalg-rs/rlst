//! Trait for LU Decomposition
pub use rlst_common::types::{IndexType, Scalar};

pub trait LUDecomp {
    type T: Scalar;

    fn into(self) -> Self;

    fn data(&self) -> &[Self::T];

    fn dim(&self) -> (IndexType, IndexType);

    fn solve(&self, vec: &mut [Self::T]);
}
