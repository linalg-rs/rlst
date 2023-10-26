//! The conjugate of a given matrix.

use crate::traits::*;
use crate::types::Scalar;
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the conjugate of a matrix.
pub type ConjugateMat<Item, MatImpl, S> = Matrix<Item, ConjugateContainer<Item, MatImpl, S>, S>;

/// A structure holding a reference to the matrix.
/// This struct implements [MatrixImplTrait] and acts like a matrix.
/// However, random access returns the corresponding conjugate entries.
pub struct ConjugateContainer<Item, MatImpl, S>(
    Matrix<Item, MatImpl, S>,
    PhantomData<Item>,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, S>;

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    ConjugateContainer<Item, MatImpl, S>
{
    pub fn new(mat: Matrix<Item, MatImpl, S>) -> Self {
        Self(mat, PhantomData, PhantomData)
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Size
    for ConjugateContainer<Item, MatImpl, S>
{
    type S = S;
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Layout
    for ConjugateContainer<Item, MatImpl, S>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> UnsafeRandomAccessByValue
    for ConjugateContainer<Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col).conj()
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index).conj()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> MatrixImplIdentifier
    for ConjugateContainer<Item, MatImpl, S>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> RawAccess
    for ConjugateContainer<Item, MatImpl, S>
{
    type T = Item;

    #[inline]
    fn data(&self) -> &[Self::T] {
        std::unimplemented!();
    }

    #[inline]
    fn get_pointer(&self) -> *const Self::T {
        std::unimplemented!();
    }

    #[inline]
    fn get_slice(&self, _first: usize, _last: usize) -> &[Self::T] {
        std::unimplemented!()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> RawAccessMut
    for ConjugateContainer<Item, MatImpl, S>
{
    #[inline]
    fn data_mut(&mut self) -> &mut [Self::T] {
        std::unimplemented!();
    }

    #[inline]
    fn get_pointer_mut(&mut self) -> *mut Self::T {
        std::unimplemented!()
    }

    #[inline]
    fn get_slice_mut(&mut self, _first: usize, _last: usize) -> &mut [Self::T] {
        std::unimplemented!()
    }
}
