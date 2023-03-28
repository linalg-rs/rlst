pub use rlst_common::types::{IndexType, Scalar};

pub trait DenseVectorInterface {
    type T: Scalar;

    fn len(&self) -> IndexType;

    unsafe fn get_unchecked(&self, elem: IndexType) -> &Self::T;
    fn get(&self, elem: IndexType) -> Option<&Self::T>;

    fn data(&self) -> &[Self::T];
}

pub trait DenseVectorInterfaceMut: DenseVectorInterface {
    unsafe fn get_unchecked_mut(&mut self, elem: IndexType) -> &mut Self::T;
    fn get_mut(&mut self, elem: IndexType) -> Option<&mut Self::T>;
    fn data_mut(&mut self) -> &mut [Self::T];
}
