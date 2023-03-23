pub use rlst_common::types::{IndexType, Scalar};

pub trait DenseMatrixAccess {
    type T: Scalar;

    fn dim(&self) -> (IndexType, IndexType);
    fn stride(&self) -> (IndexType, IndexType);

    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &Self::T;
    fn get(&self, row: IndexType, col: IndexType) -> Option<&Self::T>;

    fn data(&self) -> &[Self::T];
}

pub trait DenseMatrixAccessMut: DenseMatrixAccess {
    unsafe fn get_unchecked_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::T;
    fn get_mut(&mut self, row: IndexType, col: IndexType) -> Option<&mut Self::T>;
    fn data_mut(&mut self) -> &mut [Self::T];
}
