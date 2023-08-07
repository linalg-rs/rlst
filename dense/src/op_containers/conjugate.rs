//! The conjugate of a given matrix.

use crate::traits::*;
use crate::types::Scalar;
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the conjugate of a matrix.
pub type ConjugateMat<Item, MatImpl, RS, CS> =
    Matrix<Item, ConjugateContainer<Item, MatImpl, RS, CS>, RS, CS>;

/// A structure holding a reference to the matrix.
/// This struct implements [MatrixImplTrait] and acts like a matrix.
/// However, random access returns the corresponding conjugate entries.
pub struct ConjugateContainer<Item, MatImpl, RS, CS>(
    Matrix<Item, MatImpl, RS, CS>,
    PhantomData<Item>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, RS, CS>;

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > ConjugateContainer<Item, MatImpl, RS, CS>
{
    pub fn new(mat: Matrix<Item, MatImpl, RS, CS>) -> Self {
        Self(mat, PhantomData, PhantomData, PhantomData)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for ConjugateContainer<Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for ConjugateContainer<Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for ConjugateContainer<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixImplIdentifier for ConjugateContainer<Item, MatImpl, RS, CS>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccess for ConjugateContainer<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccessMut for ConjugateContainer<Item, MatImpl, RS, CS>
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
