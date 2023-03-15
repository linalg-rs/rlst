//! Addition of two matrices.
//!
//! This module defines a type [AdditionMat] that represents the addition of two
//! matrices. Two matrices can be added together if they have the same dimension and
//! same index layout, meaning a 1d indexing traverses both matrices in the same order.

use crate::matrix::*;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::*;

use std::marker::PhantomData;

/// A type that represents the sum of two matrices.
pub type AdditionMat<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS> =
    Matrix<Item, Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>, B, RS, CS>;

pub struct Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>(
    Matrix<Item, MatImpl1, L1, RS, CS>,
    Matrix<Item, MatImpl2, L2, RS, CS>,
    B,
    PhantomData<Item>,
    PhantomData<L1>,
    PhantomData<L2>,
    PhantomData<B>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: HScalar,
    B: BaseLayoutType,
    L1: LayoutType<IndexLayout = B>,
    L2: LayoutType<IndexLayout = B>,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl1: MatrixTrait<Item, L1, RS, CS>,
    MatImpl2: MatrixTrait<Item, L2, RS, CS>;

impl<
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>
{
    pub fn new(
        mat1: Matrix<Item, MatImpl1, L1, RS, CS>,
        mat2: Matrix<Item, MatImpl2, L2, RS, CS>,
    ) -> Self {
        assert_eq!(
            mat1.layout().dim(),
            mat2.layout().dim(),
            "Dimensions not identical in a + b with a.dim() = {:#?}, b.dim() = {:#?}",
            mat1.layout().dim(),
            mat2.layout().dim()
        );

        let layout = mat1.layout().index_layout();
        Self(
            mat1,
            mat2,
            layout,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
            PhantomData,
        )
    }
}

impl<
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>
{
    type Impl = B;

    fn layout(&self) -> &Self::Impl {
        &self.2
    }
}

impl<
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>
{
    type C = CS;
    type R = RS;
}

impl<
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccess for Addition<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(
        &self,
        row: crate::types::IndexType,
        col: crate::types::IndexType,
    ) -> Self::Item {
        self.0.get_unchecked(row, col) + self.1.get_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: crate::types::IndexType) -> Self::Item {
        self.0.get1d_unchecked(index) + self.1.get1d_unchecked(index)
    }
}

impl<
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Add<Matrix<Item, MatImpl2, L2, RS, CS>> for Matrix<Item, MatImpl1, L1, RS, CS>
{
    type Output = AdditionMat<Item, MatImpl1, MatImpl2, B, L1, L2, RS, CS>;

    fn add(self, rhs: Matrix<Item, MatImpl2, L2, RS, CS>) -> Self::Output {
        Matrix::new(Addition::new(self, rhs))
    }
}

impl<
        'a,
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Add<&'a Matrix<Item, MatImpl2, L2, RS, CS>> for Matrix<Item, MatImpl1, L1, RS, CS>
{
    type Output =
        AdditionMat<Item, MatImpl1, MatrixRef<'a, Item, MatImpl2, L2, RS, CS>, B, L1, L2, RS, CS>;

    fn add(self, rhs: &'a Matrix<Item, MatImpl2, L2, RS, CS>) -> Self::Output {
        Matrix::new(Addition::new(self, Matrix::from_ref(rhs)))
    }
}

impl<
        'a,
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Add<Matrix<Item, MatImpl2, L2, RS, CS>> for &'a Matrix<Item, MatImpl1, L1, RS, CS>
{
    type Output =
        AdditionMat<Item, MatrixRef<'a, Item, MatImpl1, L1, RS, CS>, MatImpl2, B, L1, L2, RS, CS>;

    fn add(self, rhs: Matrix<Item, MatImpl2, L2, RS, CS>) -> Self::Output {
        Matrix::new(Addition::new(Matrix::from_ref(self), rhs))
    }
}

impl<
        'a,
        Item: HScalar,
        B: BaseLayoutType,
        MatImpl1: MatrixTrait<Item, L1, RS, CS>,
        MatImpl2: MatrixTrait<Item, L2, RS, CS>,
        L1: LayoutType<IndexLayout = B>,
        L2: LayoutType<IndexLayout = B>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Add<&'a Matrix<Item, MatImpl2, L2, RS, CS>>
    for &'a Matrix<Item, MatImpl1, L1, RS, CS>
{
    type Output = AdditionMat<
        Item,
        MatrixRef<'a, Item, MatImpl1, L1, RS, CS>,
        MatrixRef<'a, Item, MatImpl2, L2, RS, CS>,
        B,
        L1,
        L2,
        RS,
        CS,
    >;

    fn add(self, rhs: &'a Matrix<Item, MatImpl2, L2, RS, CS>) -> Self::Output {
        Matrix::new(Addition::new(Matrix::from_ref(self), Matrix::from_ref(rhs)))
    }
}

#[cfg(test)]

mod test {

    use crate::layouts::RowMajor;

    use super::*;

    #[test]
    fn scalar_mult() {
        let mut mat1 = MatrixD::<f64, RowMajor>::zeros_from_dim(2, 3);
        let mut mat2 = MatrixD::<f64, RowMajor>::zeros_from_dim(2, 3);

        *mat1.get_mut(1, 2) = 5.0;
        *mat2.get_mut(1, 2) = 6.0;

        let res = 2.0 * mat1 + mat2;
        let res = res.eval();

        assert_eq!(res.get(1, 2), 16.0);
    }
}
