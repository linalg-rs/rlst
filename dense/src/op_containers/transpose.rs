//! The transpose of a given matrix.

use crate::traits::*;
use crate::types::Scalar;
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the conjugate of a matrix.
pub type TransposeMat<Item, MatImpl, S> = Matrix<Item, TransposeContainer<Item, MatImpl, S>, S>;

/// A structure holding a reference to the matrix.
/// This struct implements [MatrixImplTrait] and acts like a matrix.
/// However, random access returns the corresponding conjugate entries.
pub struct TransposeContainer<Item, MatImpl, S>(
    Matrix<Item, MatImpl, S>,
    DefaultLayout,
    PhantomData<Item>,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, S>;

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    TransposeContainer<Item, MatImpl, S>
{
    pub fn new(mat: Matrix<Item, MatImpl, S>) -> Self {
        let layout = DefaultLayout::from_dimension((mat.shape().1, mat.shape().0));
        Self(mat, layout, PhantomData, PhantomData)
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Size
    for TransposeContainer<Item, MatImpl, S>
{
    type S = S;
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> MatrixImplIdentifier
    for TransposeContainer<Item, MatImpl, S>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> RawAccess
    for TransposeContainer<Item, MatImpl, S>
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
    for TransposeContainer<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Layout
    for TransposeContainer<Item, MatImpl, S>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        &self.1
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> UnsafeRandomAccessByValue
    for TransposeContainer<Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(col, row)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        let (row, col) = self.1.convert_1d_2d(index);
        self.0.get_value_unchecked(col, row)
    }
}
