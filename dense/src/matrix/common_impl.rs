//! Implementation of common matrix traits and methods.

use crate::matrix::Matrix;
use crate::matrix_view::{MatrixView, MatrixViewMut};
use crate::types::Scalar;
use crate::{traits::*, DefaultLayout, ViewMatrixMut};
use crate::{MatrixD, RefMat};
use crate::{RefMatMut, ViewMatrix};
use num::traits::Zero;
use rlst_common::traits::*;

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for Matrix<Item, MatImpl, RS, CS>
{
    type Impl = DefaultLayout;
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Shape for Matrix<Item, MatImpl, RS, CS>
{
    fn shape(&self) -> (usize, usize) {
        self.layout().dim()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Stride for Matrix<Item, MatImpl, RS, CS>
{
    fn stride(&self) -> (usize, usize) {
        self.layout().stride()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > NumberOfElements for Matrix<Item, MatImpl, RS, CS>
{
    fn number_of_elements(&self) -> usize {
        self.layout().number_of_elements()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for Matrix<Item, MatImpl, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for Matrix<Item, MatImpl, RS, CS>
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
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessMut for Matrix<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitAccessByRef<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByRef for Matrix<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitAccessByRef<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Index<[usize; 2]> for Matrix<Item, MatImpl, RS, CS>
{
    type Output = Item;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        self.get(index[0], index[1]).unwrap()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS> + MatrixImplTraitAccessByRef<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::IndexMut<[usize; 2]> for Matrix<Item, MatImpl, RS, CS>
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.get_mut(index[0], index[1]).unwrap()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Eval for Matrix<Item, MatImpl, RS, CS>
where
    Self: NewLikeSelf,
    <Self as NewLikeSelf>::Out: ColumnMajorIteratorMut<T = Item>,
{
    type Out = <Self as NewLikeSelf>::Out;

    fn eval(&self) -> Self::Out {
        let mut result = self.new_like_self();
        for (res, value) in result.iter_col_major_mut().zip(self.iter_col_major()) {
            *res = value;
        }
        result
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Copy for Matrix<Item, MatImpl, RS, CS>
where
    Self: Eval,
{
    type Out = <Self as Eval>::Out;

    fn copy(&self) -> Self::Out {
        self.eval()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > ForEach for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;
    fn for_each<F: FnMut(&mut Self::T)>(&mut self, mut f: F) {
        for index in 0..self.layout().number_of_elements() {
            f(self.get1d_mut(index).unwrap())
        }
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS> + RawAccess<T = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccess for Matrix<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS> + RawAccessMut<T = Item>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > RawAccessMut for Matrix<Item, MatImpl, RS, CS>
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

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > ScaleInPlace for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;

    fn scale_in_place(&mut self, alpha: Self::T) {
        self.for_each(|elem| *elem = alpha * *elem);
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Other: Shape + ColumnMajorIterator<T = Item>,
    > FillFrom<Other> for Matrix<Item, MatImpl, RS, CS>
{
    fn fill_from(&mut self, other: &Other) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shapes do not agree. {:#?} != {:#?}",
            self.shape(),
            other.shape()
        );

        for (item, other_item) in self.iter_col_major_mut().zip(other.iter_col_major()) {
            *item = other_item
        }
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Other: Shape + ColumnMajorIterator<T = Item>,
    > SumInto<Other> for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;

    fn sum_into(&mut self, alpha: Self::T, other: &Other) {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shapes do not agree. {:#?} != {:#?}",
            self.shape(),
            other.shape()
        );

        for (item, other_item) in self.iter_col_major_mut().zip(other.iter_col_major()) {
            *item += alpha * other_item
        }
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SetDiag for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;

    fn set_diag_from_iter<I: Iterator<Item = Self::T>>(&mut self, iter: I) {
        for (item, other) in self.iter_diag_mut().zip(iter) {
            *item = other;
        }
    }

    fn set_diag_from_slice(&mut self, diag: &[Self::T]) {
        let k = std::cmp::min(self.shape().0, self.shape().1);

        assert_eq!(
            k,
            diag.len(),
            "Expected length of slice {} but actual length is {}",
            k,
            diag.len()
        );

        self.set_diag_from_iter(diag.iter().copied());
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SquareSum for Matrix<Item, MatImpl, RS, CS>
{
    type T = Item;
    fn square_sum(&self) -> <Self::T as Scalar>::Real {
        let mut result = <<Self::T as Scalar>::Real as Zero>::zero();
        for index in 0..self.number_of_elements() {
            let value = self.get1d_value(index).unwrap();
            result += value.square();
        }
        result
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, RS, CS>
{
    /// Return a view to the whole matrix.
    pub fn view(&self) -> RefMat<Item, MatImpl, RS, CS> {
        Matrix::from_ref(self)
    }

    /// Return a view onto a subblock of the matrix.
    pub fn subview(
        &self,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> ViewMatrix<Item, MatImpl, RS, CS> {
        Matrix::new(MatrixView::new(self, offset, block_size))
    }

    /// Return a single column of a matrix.
    pub fn col(&self, col_index: usize) -> ViewMatrix<Item, MatImpl, RS, CS> {
        self.subview((0, col_index), (self.shape().0, 1))
    }

    /// Return a single row of a matrix.
    pub fn row(&self, row_index: usize) -> ViewMatrix<Item, MatImpl, RS, CS> {
        self.subview((row_index, 0), (1, self.shape().1))
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, RS, CS>
{
    /// Return a mutable view to the whole matrix.
    pub fn view_mut(&mut self) -> RefMatMut<Item, MatImpl, RS, CS> {
        Matrix::from_ref_mut(self)
    }

    /// Return a mutable view onto a subblock of the matrix.
    pub fn subview_mut(
        &mut self,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> ViewMatrixMut<Item, MatImpl, RS, CS> {
        Matrix::new(MatrixViewMut::new(self, offset, block_size))
    }

    /// Return a mutable single column of a matrix.
    pub fn col_mut(&mut self, col_index: usize) -> ViewMatrixMut<Item, MatImpl, RS, CS> {
        self.subview_mut((0, col_index), (self.shape().0, 1))
    }

    /// Return a mutable single row of a matrix.
    pub fn row_mut(&mut self, row_index: usize) -> ViewMatrixMut<Item, MatImpl, RS, CS> {
        self.subview_mut((row_index, 0), (1, self.shape().1))
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > IsHermitian for Matrix<Item, MatImpl, RS, CS>
{
    fn is_hermitian(&self) -> bool {
        if self.shape().0 != self.shape().1 {
            return false;
        }

        let mut hermitian = true;

        'outer: for col in 0..self.shape().1 {
            for row in col..self.shape().0 {
                if self.get_value(row, col).unwrap() != self.get_value(col, row).unwrap().conj() {
                    hermitian = false;
                    break 'outer;
                }
            }
        }
        hermitian
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > IsSymmetric for Matrix<Item, MatImpl, RS, CS>
{
    fn is_symmetric(&self) -> bool {
        if self.shape().0 != self.shape().1 {
            return false;
        }

        let mut symmetric = true;

        'outer: for col in 0..self.shape().1 {
            for row in col..self.shape().0 {
                if self.get_value(row, col).unwrap() != self.get_value(col, row).unwrap() {
                    symmetric = false;
                    break 'outer;
                }
            }
        }
        symmetric
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, RS, CS>
{
    pub fn get_mat_impl_type(&self) -> MatrixImplType {
        MatImpl::MAT_IMPL
    }

    pub fn to_dyn_matrix(&self) -> MatrixD<Item> {
        let mut mat = crate::rlst_mat![Item, self.shape()];

        for (item, other) in mat.iter_col_major_mut().zip(self.iter_col_major()) {
            *item = other;
        }
        mat
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, RS, CS>
{
    pub fn sum_into_block<
        Other: RandomAccessByValue<Item = Item> + ColumnMajorIterator<T = Item> + Shape,
    >(
        &mut self,
        alpha: Item,
        top_left: (usize, usize),
        other: &Other,
    ) {
        let mut subview = self.subview_mut(top_left, other.shape());
        subview.sum_into(alpha, other);
    }
}
