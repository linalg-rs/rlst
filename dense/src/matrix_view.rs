//! A matrix view is a new matrix that is a view onto a subpart of an existing matrix.
//!
//! A matrix view does not have its own data slice. It adapts its random access routines
//! to point to a specified block of another matrix.

use crate::matrix::Matrix;
use crate::types::Scalar;
use crate::{traits::*, RefMat};
use crate::{DefaultLayout, RefMatMut};

pub struct MatrixView<'a, Item, MatImpl, RS, CS>
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, RS, CS>,
{
    mat: RefMat<'a, Item, MatImpl, RS, CS>,
    offset: (usize, usize),
    layout: DefaultLayout,
}

pub struct MatrixViewMut<'a, Item, MatImpl, RS, CS>
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixImplTraitMut<Item, RS, CS>,
{
    mat: RefMatMut<'a, Item, MatImpl, RS, CS>,
    offset: (usize, usize),
    layout: DefaultLayout,
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixView<'a, Item, MatImpl, RS, CS>
{
    pub fn new(
        mat: &'a Matrix<Item, MatImpl, RS, CS>,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> Self {
        assert!(
            offset.0 + block_size.0 <= mat.shape().0 && offset.1 + block_size.1 <= mat.shape().1,
            "Block exceeds matrix dimensions. offset: {:#?}, block size: {:#?}",
            offset,
            block_size
        );

        let layout = DefaultLayout::column_major(block_size);

        Self {
            mat: mat.view(),
            offset,
            layout,
        }
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    pub fn new(
        mat: &'a mut Matrix<Item, MatImpl, RS, CS>,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> Self {
        assert!(
            offset.0 + block_size.0 <= mat.shape().0 && offset.1 + block_size.1 <= mat.shape().1,
            "Block exceeds matrix dimensions. offset: {:#?}, block size: {:#?}",
            offset,
            block_size
        );

        let layout = DefaultLayout::column_major(block_size);

        Self {
            mat: mat.view_mut(),
            offset,
            layout,
        }
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for MatrixView<'a, Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        &self.layout
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        &self.layout
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for MatrixView<'a, Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixImplIdentifier for MatrixView<'a, Item, MatImpl, RS, CS>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > MatrixImplIdentifier for MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Derived;
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for MatrixView<'a, Item, MatImpl, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.mat
            .get_value_unchecked(self.offset.0 + row, self.offset.1 + col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        let val_2d = self.layout.convert_1d_2d(index);
        self.get_value_unchecked(val_2d.0, val_2d.1)
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.mat
            .get_value_unchecked(self.offset.0 + row, self.offset.1 + col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        let val_2d = self.layout.convert_1d_2d(index);
        self.get_value_unchecked(val_2d.0, val_2d.1)
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitAccessByRef<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByRef for MatrixView<'a, Item, MatImpl, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item {
        self.mat
            .get_unchecked(self.offset.0 + row, self.offset.1 + col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item {
        let val_2d = self.layout.convert_1d_2d(index);
        self.get_unchecked(val_2d.0, val_2d.1)
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitAccessByRef<Item, RS, CS> + MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByRef for MatrixViewMut<'a, Item, MatImpl, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item {
        self.mat
            .get_unchecked(self.offset.0 + row, self.offset.1 + col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item {
        let val_2d = self.layout.convert_1d_2d(index);
        self.get_unchecked(val_2d.0, val_2d.1)
    }
}

impl<
        'a,
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessMut for MatrixViewMut<'a, Item, MatImpl, RS, CS>
where
    Matrix<Item, MatImpl, RS, CS>: UnsafeRandomAccessMut<Item = Item>,
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Item {
        self.mat
            .get_unchecked_mut(self.offset.0 + row, self.offset.1 + col)
    }

    #[inline]
    unsafe fn get1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        let val_2d = self.layout.convert_1d_2d(index);
        self.get_unchecked_mut(val_2d.0, val_2d.1)
    }
}

#[cfg(test)]
mod test {

    use rlst_common::traits::*;

    #[test]
    fn test_col() {
        let mat = crate::rlst_rand_mat![f64, (6, 5)];
        let col = mat.col(4);

        assert_eq!(col.shape(), (6, 1));

        for (index, col_item) in col.iter_col_major().enumerate() {
            assert_eq![col_item, mat[[index, 4]]];
        }
    }

    #[test]
    fn test_set_element() {
        let mut mat = crate::rlst_rand_mat![f64, (6, 5)];
        let mut col = mat.col_mut(4);

        assert_eq!(col.shape(), (6, 1));

        col[[0, 0]] = 5.0;
        assert_eq![mat[[0, 4]], 5.0];
    }

    #[test]
    fn test_row() {
        let mat = crate::rlst_rand_mat![f64, (6, 5)];
        let row = mat.row(3);

        assert_eq!(row.shape(), (1, 5));

        for (index, row_item) in row.iter_col_major().enumerate() {
            assert_eq![row_item, mat[[3, index]]];
        }
    }

    #[test]
    fn test_iterated_view() {
        let mat = crate::rlst_rand_mat![f64, (6, 5)];
        let block1 = mat.subview((1, 1), (2, 3));
        let col = block1.col(1);

        for (index, col_item) in col.iter_col_major().enumerate() {
            assert_eq![col_item, mat[[1 + index, 2]]];
        }
    }

    #[test]
    fn test_iterated_view_mut() {
        let mut mat = crate::rlst_rand_mat![f64, (6, 5)];
        let mut block1 = mat.subview_mut((1, 1), (2, 3));
        let mut col = block1.col_mut(1);

        col[[1, 0]] = 3.0;

        assert_eq![mat[[2, 2]], 3.0];
    }
}
