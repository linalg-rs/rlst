//! Componentwise product of two matrices.
//!
//! This module implements the Hadamard product of two
//! matrices. Instead of executing the Hadamard product immediately,
//! an operator is created that stores the original matrices and
//! whose element accessor routines return the entries of the Hadamard product.

use crate::matrix::*;
use crate::traits::*;
use crate::types::*;
use crate::DefaultLayout;

use std::marker::PhantomData;

/// A type that represents the componentwise product of two matrices.
pub type CmpWiseProductMat<Item, MatImpl1, MatImpl2, S> =
    Matrix<Item, CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>, S>;

pub struct CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>(
    Matrix<Item, MatImpl1, S>,
    Matrix<Item, MatImpl2, S>,
    DefaultLayout,
    PhantomData<S>,
)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl1: MatrixImplTrait<Item, S>,
    MatImpl2: MatrixImplTrait<Item, S>;

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
{
    pub fn new(mat1: Matrix<Item, MatImpl1, S>, mat2: Matrix<Item, MatImpl2, S>) -> Self {
        assert_eq!(
            mat1.layout().dim(),
            mat2.layout().dim(),
            "Dimensions not identical in Hadamard product of a and b with a.dim() = {:#?}, b.dim() = {:#?}",
            mat1.layout().dim(),
            mat2.layout().dim()
        );
        let dim = mat1.layout().dim();
        Self(mat1, mat2, DefaultLayout::from_dimension(dim), PhantomData)
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > Layout for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
{
    type Impl = DefaultLayout;

    fn layout(&self) -> &Self::Impl {
        &self.2
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > Size for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
{
    type S = S;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > UnsafeRandomAccessByValue for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col) * self.1.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index) * self.1.get1d_value_unchecked(index)
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > MatrixImplIdentifier for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > RawAccess for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
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
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > RawAccessMut for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, S>
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
