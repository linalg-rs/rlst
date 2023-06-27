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
pub type CmpWiseProductMat<Item, MatImpl1, MatImpl2, RS, CS> =
    Matrix<Item, CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>, RS, CS>;

pub struct CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>(
    Matrix<Item, MatImpl1, RS, CS>,
    Matrix<Item, MatImpl2, RS, CS>,
    DefaultLayout,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl1: MatrixImplTrait<Item, RS, CS>,
    MatImpl2: MatrixImplTrait<Item, RS, CS>;

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>
{
    pub fn new(mat1: Matrix<Item, MatImpl1, RS, CS>, mat2: Matrix<Item, MatImpl2, RS, CS>) -> Self {
        assert_eq!(
            mat1.layout().dim(),
            mat2.layout().dim(),
            "Dimensions not identical in Hadamard product of a and b with a.dim() = {:#?}, b.dim() = {:#?}",
            mat1.layout().dim(),
            mat2.layout().dim()
        );
        let dim = mat1.layout().dim();
        Self(
            mat1,
            mat2,
            DefaultLayout::from_dimension(dim),
            PhantomData,
            PhantomData,
        )
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>
{
    type Impl = DefaultLayout;

    fn layout(&self) -> &Self::Impl {
        &self.2
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>
{
    type C = CS;
    type R = RS;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for CmpWiseProductContainer<Item, MatImpl1, MatImpl2, RS, CS>
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
