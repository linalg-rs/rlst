//! Addition of two matrices.
//!
//! This module defines a type [SubtractionMat] that represents the subtraction of two
//! matrices. Two matrices can be subtracted if they have the same dimension and
//! same index layout, meaning a 1d indexing traverses both matrices in the same order.

use crate::matrix::*;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::*;
use crate::DefaultLayout;

use std::marker::PhantomData;

/// A type that represents the sum of two matrices.
pub type SubtractionMat<Item, MatImpl1, MatImpl2, RS, CS> =
    Matrix<Item, Subtraction<Item, MatImpl1, MatImpl2, RS, CS>, RS, CS>;

pub struct Subtraction<Item, MatImpl1, MatImpl2, RS, CS>(
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
    > Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
{
    pub fn new(mat1: Matrix<Item, MatImpl1, RS, CS>, mat2: Matrix<Item, MatImpl2, RS, CS>) -> Self {
        assert_eq!(
            mat1.layout().dim(),
            mat2.layout().dim(),
            "Dimensions not identical in a + b with a.dim() = {:#?}, b.dim() = {:#?}",
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
    > Layout for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
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
    > SizeType for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
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
    > MatrixImplIdentifier for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccess for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
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
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccessMut for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for Subtraction<Item, MatImpl1, MatImpl2, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col) - self.1.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index) - self.1.get1d_value_unchecked(index)
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Sub<Matrix<Item, MatImpl2, RS, CS>> for Matrix<Item, MatImpl1, RS, CS>
{
    type Output = SubtractionMat<Item, MatImpl1, MatImpl2, RS, CS>;

    fn sub(self, rhs: Matrix<Item, MatImpl2, RS, CS>) -> Self::Output {
        Matrix::new(Subtraction::new(self, rhs))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Sub<&'a Matrix<Item, MatImpl2, RS, CS>> for Matrix<Item, MatImpl1, RS, CS>
{
    type Output = SubtractionMat<Item, MatImpl1, MatrixRef<'a, Item, MatImpl2, RS, CS>, RS, CS>;

    fn sub(self, rhs: &'a Matrix<Item, MatImpl2, RS, CS>) -> Self::Output {
        Matrix::new(Subtraction::new(self, Matrix::from_ref(rhs)))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Sub<Matrix<Item, MatImpl2, RS, CS>> for &'a Matrix<Item, MatImpl1, RS, CS>
{
    type Output = SubtractionMat<Item, MatrixRef<'a, Item, MatImpl1, RS, CS>, MatImpl2, RS, CS>;

    fn sub(self, rhs: Matrix<Item, MatImpl2, RS, CS>) -> Self::Output {
        Matrix::new(Subtraction::new(Matrix::from_ref(self), rhs))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Sub<&'a Matrix<Item, MatImpl2, RS, CS>> for &'a Matrix<Item, MatImpl1, RS, CS>
{
    type Output = SubtractionMat<
        Item,
        MatrixRef<'a, Item, MatImpl1, RS, CS>,
        MatrixRef<'a, Item, MatImpl2, RS, CS>,
        RS,
        CS,
    >;

    fn sub(self, rhs: &'a Matrix<Item, MatImpl2, RS, CS>) -> Self::Output {
        Matrix::new(Subtraction::new(
            Matrix::from_ref(self),
            Matrix::from_ref(rhs),
        ))
    }
}

#[cfg(test)]

mod test {

    use rlst_common::traits::*;

    #[test]
    fn scalar_mult() {
        let mut mat1 = crate::rlst_mat![f64, (2, 3)];
        let mut mat2 = crate::rlst_mat![f64, (2, 3)];

        mat1[[1, 2]] = 5.0;
        mat2[[1, 2]] = 6.0;

        let res = 2.0 * mat1 + mat2;
        let res = res.eval();

        assert_eq!(res[[1, 2]], 16.0);
    }
}
