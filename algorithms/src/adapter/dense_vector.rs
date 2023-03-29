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

/// A simple dense vector interface type.
pub struct DenseVector<VecImpl: DenseVectorInterface> {
    vec: VecImpl,
}

pub struct VecFromSlice<'a, T: Scalar> {
    slice: &'a [T],
}

pub struct VecFromSliceMut<'a, T: Scalar> {
    slice: &'a mut [T],
}

impl<VecImpl: DenseVectorInterface> DenseVector<VecImpl> {
    pub fn len(&self) -> IndexType {
        self.vec.len()
    }

    pub fn data(&self) -> &[VecImpl::T] {
        self.vec.data()
    }

    pub fn get(&self, elem: IndexType) -> Option<&VecImpl::T> {
        self.vec.get(elem)
    }

    pub unsafe fn get_unchecked(&self, elem: IndexType) -> &VecImpl::T {
        self.vec.get_unchecked(elem)
    }
}

impl<VecImpl: DenseVectorInterfaceMut> DenseVector<VecImpl> {
    pub fn data_mut(&mut self) -> &mut [VecImpl::T] {
        self.vec.data_mut()
    }

    pub fn get_mut(&mut self, elem: IndexType) -> Option<&mut VecImpl::T> {
        self.vec.get_mut(elem)
    }

    pub unsafe fn get_unchecked_mut(&mut self, elem: IndexType) -> &mut VecImpl::T {
        self.vec.get_unchecked_mut(elem)
    }
}

impl<'a, T: Scalar> DenseVectorInterface for VecFromSlice<'a, T> {
    type T = T;

    fn len(&self) -> IndexType {
        self.slice.len()
    }

    fn data(&self) -> &[Self::T] {
        self.slice
    }

    fn get(&self, elem: IndexType) -> Option<&Self::T> {
        self.slice.get(elem as usize)
    }

    unsafe fn get_unchecked(&self, elem: IndexType) -> &Self::T {
        self.slice.get_unchecked(elem)
    }
}

impl<'a, T: Scalar> DenseVectorInterface for VecFromSliceMut<'a, T> {
    type T = T;

    fn len(&self) -> IndexType {
        self.slice.len()
    }

    fn data(&self) -> &[Self::T] {
        self.slice
    }

    fn get(&self, elem: IndexType) -> Option<&Self::T> {
        self.slice.get(elem as usize)
    }

    unsafe fn get_unchecked(&self, elem: IndexType) -> &Self::T {
        self.slice.get_unchecked(elem)
    }
}

impl<'a, T: Scalar> DenseVectorInterfaceMut for VecFromSliceMut<'a, T> {
    fn data_mut(&mut self) -> &mut [Self::T] {
        self.slice
    }

    fn get_mut(&mut self, elem: IndexType) -> Option<&mut Self::T> {
        self.slice.get_mut(elem)
    }

    unsafe fn get_unchecked_mut(&mut self, elem: IndexType) -> &mut Self::T {
        self.slice.get_unchecked_mut(elem)
    }
}
