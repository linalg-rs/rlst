//! Addition of two matrices.
//!
//! This module defines a type [AdditionMat] that represents the addition of two
//! matrices.

use crate::matrix::*;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::*;
use crate::DefaultLayout;

use std::marker::PhantomData;

/// A type that represents the sum of two matrices.
pub type AdditionMat<Item, MatImpl1, MatImpl2, S> =
    Matrix<Item, Addition<Item, MatImpl1, MatImpl2, S>, S>;

pub struct Addition<Item, MatImpl1, MatImpl2, S>(
    Matrix<Item, MatImpl1, S>,
    Matrix<Item, MatImpl2, S>,
    DefaultLayout,
    PhantomData<S>,
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
    > Addition<Item, MatImpl1, MatImpl2, S>
{
    pub fn new(mat1: Matrix<Item, MatImpl1, S>, mat2: Matrix<Item, MatImpl2, S>) -> Self {
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
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > Layout for Addition<Item, MatImpl1, MatImpl2, S>
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
    > Size for Addition<Item, MatImpl1, MatImpl2, S>
{
    type S = S;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > UnsafeRandomAccessByValue for Addition<Item, MatImpl1, MatImpl2, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.0.get_value_unchecked(row, col) + self.1.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.0.get1d_value_unchecked(index) + self.1.get1d_value_unchecked(index)
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > MatrixImplIdentifier for Addition<Item, MatImpl1, MatImpl2, S>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > RawAccess for Addition<Item, MatImpl1, MatImpl2, S>
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
    > RawAccessMut for Addition<Item, MatImpl1, MatImpl2, S>
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
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > std::ops::Add<Matrix<Item, MatImpl2, S>> for Matrix<Item, MatImpl1, S>
{
    type Output = AdditionMat<Item, MatImpl1, MatImpl2, S>;

    fn add(self, rhs: Matrix<Item, MatImpl2, S>) -> Self::Output {
        Matrix::new(Addition::new(self, rhs))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > std::ops::Add<&'a Matrix<Item, MatImpl2, S>> for Matrix<Item, MatImpl1, S>
{
    type Output = AdditionMat<Item, MatImpl1, MatrixRef<'a, Item, MatImpl2, S>, S>;

    fn add(self, rhs: &'a Matrix<Item, MatImpl2, S>) -> Self::Output {
        Matrix::new(Addition::new(self, Matrix::from_ref(rhs)))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > std::ops::Add<Matrix<Item, MatImpl2, S>> for &'a Matrix<Item, MatImpl1, S>
{
    type Output = AdditionMat<Item, MatrixRef<'a, Item, MatImpl1, S>, MatImpl2, S>;

    fn add(self, rhs: Matrix<Item, MatImpl2, S>) -> Self::Output {
        Matrix::new(Addition::new(Matrix::from_ref(self), rhs))
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > std::ops::Add<&'a Matrix<Item, MatImpl2, S>> for &'a Matrix<Item, MatImpl1, S>
{
    type Output =
        AdditionMat<Item, MatrixRef<'a, Item, MatImpl1, S>, MatrixRef<'a, Item, MatImpl2, S>, S>;

    fn add(self, rhs: &'a Matrix<Item, MatImpl2, S>) -> Self::Output {
        Matrix::new(Addition::new(Matrix::from_ref(self), Matrix::from_ref(rhs)))
    }
}

#[cfg(test)]

mod test {

    use rlst_common::traits::*;

    #[test]
    fn scalar_mult() {
        let mut mat1 = crate::rlst_mat!(f64, (2, 3));
        let mut mat2 = crate::rlst_mat!(f64, (2, 3));

        mat1[[1, 2]] = 5.0;
        mat2[[1, 2]] = 6.0;

        let res = 2.0 * mat1 + mat2;
        let res = res.eval();

        assert_eq!(res[[1, 2]], 16.0);
    }
}
