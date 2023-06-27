//! The transpose of a given matrix.

use crate::traits::*;
use crate::types::Scalar;
use crate::{matrix::*, DefaultLayout};
use std::marker::PhantomData;

/// This type represents the conjugate of a matrix.
pub type TransposeMat<Item, MatImpl, RS, CS> =
    Matrix<Item, TransposeContainer<Item, MatImpl, RS, CS>, RS, CS>;

/// A structure holding a reference to the matrix.
/// This struct implements [MatrixImplTrait] and acts like a matrix.
/// However, random access returns the corresponding conjugate entries.
pub struct TransposeContainer<Item, MatImpl, RS, CS>(
    Matrix<Item, MatImpl, RS, CS>,
    DefaultLayout,
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
    > TransposeContainer<Item, MatImpl, RS, CS>
{
    pub fn new(mat: Matrix<Item, MatImpl, RS, CS>) -> Self {
        let layout = DefaultLayout::from_dimension((mat.shape().1, mat.shape().0));
        Self(mat, layout, PhantomData, PhantomData, PhantomData)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for TransposeContainer<Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for TransposeContainer<Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        &self.1
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for TransposeContainer<Item, MatImpl, RS, CS>
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
