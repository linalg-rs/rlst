//! A matrix that holds a reference to another matrix.
//!
//! The type defined in this module translates a reference to a matrix into an owned matrix.
//! Why is this necessary? Consider the code `let sum = mat1 + mat2` with `mat1` and `mat2`
//! matrices. The result `sum` is an addition type that now owns `mat1` and `mat2`. But
//! often we do not want `sum` to take ownership of the terms on the right hand side.
//! So we want to write `let sum = &mat1 + &mat2`. But now the type that implements addition
//! needs to hold references to matrices and cannot take ownership anymore. So we need to
//! implement different addition types for each of the cases `mat1 + mat2`, `&mat1 + mat2`,
//! `mat1 + &mat2`, `&mat2 + &mat2`. This would require significant code duplication. The
//! solution is to create a type that turns a reference to a matrix into an owned matrix.
//! This is what [MatrixRef] is doing. It simply takes a reference to a matrix and forwards
//! all matrix operations to the reference. Hence, an expression of the form `&mat1 + mat2` will
//! first be converted into an expression similar to `MatrixRef(&mat1) + mat2`, and then
//! both terms passed onto the addition type, which takes ownership of both terms.

use crate::matrix::Matrix;
use crate::traits::*;
use crate::types::Scalar;
use crate::DefaultLayout;
use std::marker::PhantomData;

/// A struct that implements [MatrixImplTrait] by holding a reference
/// to a matrix and forwarding all matrix operations to the held reference.
pub struct MatrixRef<'a, Item, MatImpl, S>(
    &'a Matrix<Item, MatImpl, S>,
    PhantomData<Item>,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, S>;

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    MatrixRef<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a Matrix<Item, MatImpl, S>) -> Self {
        Self(mat, PhantomData, PhantomData)
    }
}

pub struct MatrixRefMut<'a, Item, MatImpl, S>(
    &'a mut Matrix<Item, MatImpl, S>,
    PhantomData<Item>,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTraitMut<Item, S>;

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    MatrixRefMut<'a, Item, MatImpl, S>
{
    pub fn new(mat: &'a mut Matrix<Item, MatImpl, S>) -> Self {
        Self(mat, PhantomData, PhantomData)
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Layout
    for MatrixRef<'a, Item, MatImpl, S>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Size
    for MatrixRef<'a, Item, MatImpl, S>
{
    type S = S;
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> MatrixImplIdentifier
    for MatrixRef<'a, Item, MatImpl, S>
{
    const MAT_IMPL: MatrixImplType = MatImpl::MAT_IMPL;
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier>
    UnsafeRandomAccessByValue for MatrixRef<'a, Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index)
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> Layout
    for MatrixRefMut<'a, Item, MatImpl, S>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> MatrixImplIdentifier
    for MatrixRefMut<'a, Item, MatImpl, S>
{
    const MAT_IMPL: MatrixImplType = MatImpl::MAT_IMPL;
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> Size
    for MatrixRefMut<'a, Item, MatImpl, S>
{
    type S = S;
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    UnsafeRandomAccessByValue for MatrixRefMut<'a, Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index)
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, S> + MatrixImplTraitAccessByRef<Item, S>,
        S: SizeIdentifier,
    > UnsafeRandomAccessByRef for MatrixRefMut<'a, Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item {
        self.0.get_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get1d_unchecked(index)
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitAccessByRef<Item, S>, S: SizeIdentifier>
    UnsafeRandomAccessByRef for MatrixRef<'a, Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item {
        self.0.get_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get1d_unchecked(index)
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    UnsafeRandomAccessMut for MatrixRefMut<'a, Item, MatImpl, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Item {
        self.0.get_unchecked_mut(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get1d_unchecked_mut(index)
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> RawAccess
    for MatrixRefMut<'a, Item, MatImpl, S>
where
    Matrix<Item, MatImpl, S>: RawAccess<T = Item>,
{
    type T = Item;

    #[inline]
    fn get_pointer(&self) -> *const Item {
        self.0.get_pointer()
    }

    #[inline]
    fn get_slice(&self, first: usize, last: usize) -> &[Item] {
        self.0.get_slice(first, last)
    }

    #[inline]
    fn data(&self) -> &[Item] {
        self.0.data()
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> RawAccess
    for MatrixRef<'a, Item, MatImpl, S>
where
    Matrix<Item, MatImpl, S>: RawAccess<T = Item>,
{
    type T = Item;

    #[inline]
    fn get_pointer(&self) -> *const Item {
        self.0.get_pointer()
    }

    #[inline]
    fn get_slice(&self, first: usize, last: usize) -> &[Item] {
        self.0.get_slice(first, last)
    }

    #[inline]
    fn data(&self) -> &[Item] {
        self.0.data()
    }
}

impl<'a, Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> RawAccessMut
    for MatrixRefMut<'a, Item, MatImpl, S>
where
    Matrix<Item, MatImpl, S>: RawAccessMut<T = Item>,
{
    fn get_pointer_mut(&mut self) -> *mut Item {
        self.0.get_pointer_mut()
    }

    fn get_slice_mut(&mut self, first: usize, last: usize) -> &mut [Item] {
        self.0.get_slice_mut(first, last)
    }

    fn data_mut(&mut self) -> &mut [Item] {
        self.0.data_mut()
    }
}
