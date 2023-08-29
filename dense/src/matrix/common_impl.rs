//! Implementation of common matrix traits and methods.

use crate::matrix::Matrix;
use crate::matrix_view::{MatrixView, MatrixViewMut};
use crate::types::Scalar;
use crate::{traits::*, DefaultLayout, ViewMatrixMut};
use crate::{MatrixD, RefMat};
use crate::{RefMatMut, ViewMatrix};
use num::traits::Zero;
use rlst_common::traits::*;

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Layout
    for Matrix<Item, MatImpl, S>
{
    type Impl = DefaultLayout;
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Shape
    for Matrix<Item, MatImpl, S>
{
    fn shape(&self) -> (usize, usize) {
        self.layout().dim()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Stride
    for Matrix<Item, MatImpl, S>
{
    fn stride(&self) -> (usize, usize) {
        self.layout().stride()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> NumberOfElements
    for Matrix<Item, MatImpl, S>
{
    fn number_of_elements(&self) -> usize {
        self.layout().number_of_elements()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Size
    for Matrix<Item, MatImpl, S>
{
    type S = S;
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> UnsafeRandomAccessByValue
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> UnsafeRandomAccessMut
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTraitAccessByRef<Item, S>, S: SizeIdentifier>
    UnsafeRandomAccessByRef for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTraitAccessByRef<Item, S>, S: SizeIdentifier>
    std::ops::Index<[usize; 2]> for Matrix<Item, MatImpl, S>
{
    type Output = Item;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        self.get(index[0], index[1]).unwrap()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, S> + MatrixImplTraitAccessByRef<Item, S>,
        S: SizeIdentifier,
    > std::ops::IndexMut<[usize; 2]> for Matrix<Item, MatImpl, S>
{
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        self.get_mut(index[0], index[1]).unwrap()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Eval
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Copy
    for Matrix<Item, MatImpl, S>
where
    Self: Eval,
{
    type Out = <Self as Eval>::Out;

    fn copy(&self) -> Self::Out {
        self.eval()
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> ForEach
    for Matrix<Item, MatImpl, S>
{
    type T = Item;
    fn for_each<F: FnMut(&mut Self::T)>(&mut self, mut f: F) {
        for index in 0..self.layout().number_of_elements() {
            f(self.get1d_mut(index).unwrap())
        }
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S> + RawAccess<T = Item>, S: SizeIdentifier>
    RawAccess for Matrix<Item, MatImpl, S>
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
        MatImpl: MatrixImplTraitMut<Item, S> + RawAccessMut<T = Item>,
        S: SizeIdentifier,
    > RawAccessMut for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> ScaleInPlace
    for Matrix<Item, MatImpl, S>
{
    type T = Item;

    fn scale_in_place(&mut self, alpha: Self::T) {
        self.for_each(|elem| *elem = alpha * *elem);
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, S>,
        S: SizeIdentifier,
        Other: Shape + ColumnMajorIterator<T = Item>,
    > FillFrom<Other> for Matrix<Item, MatImpl, S>
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
        MatImpl: MatrixImplTraitMut<Item, S>,
        S: SizeIdentifier,
        Other: Shape + ColumnMajorIterator<T = Item>,
    > SumInto<Other> for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier> SetDiag
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> SquareSum
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Matrix<Item, MatImpl, S> {
    /// Return a view to the whole matrix.
    pub fn view(&self) -> RefMat<Item, MatImpl, S> {
        Matrix::from_ref(self)
    }

    /// Return a view onto a subblock of the matrix.
    pub fn subview(
        &self,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> ViewMatrix<Item, MatImpl, S> {
        Matrix::new(MatrixView::new(self, offset, block_size))
    }

    /// Return a single column of a matrix.
    pub fn col(&self, col_index: usize) -> ViewMatrix<Item, MatImpl, S> {
        self.subview((0, col_index), (self.shape().0, 1))
    }

    /// Return a single row of a matrix.
    pub fn row(&self, row_index: usize) -> ViewMatrix<Item, MatImpl, S> {
        self.subview((row_index, 0), (1, self.shape().1))
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    Matrix<Item, MatImpl, S>
{
    /// Return a mutable view to the whole matrix.
    pub fn view_mut(&mut self) -> RefMatMut<Item, MatImpl, S> {
        Matrix::from_ref_mut(self)
    }

    /// Return a mutable view onto a subblock of the matrix.
    pub fn subview_mut(
        &mut self,
        offset: (usize, usize),
        block_size: (usize, usize),
    ) -> ViewMatrixMut<Item, MatImpl, S> {
        Matrix::new(MatrixViewMut::new(self, offset, block_size))
    }

    /// Return a mutable single column of a matrix.
    pub fn col_mut(&mut self, col_index: usize) -> ViewMatrixMut<Item, MatImpl, S> {
        self.subview_mut((0, col_index), (self.shape().0, 1))
    }

    /// Return a mutable single row of a matrix.
    pub fn row_mut(&mut self, row_index: usize) -> ViewMatrixMut<Item, MatImpl, S> {
        self.subview_mut((row_index, 0), (1, self.shape().1))
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> IsHermitian
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> IsSymmetric
    for Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Matrix<Item, MatImpl, S> {
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

impl<Item: Scalar, MatImpl: MatrixImplTraitMut<Item, S>, S: SizeIdentifier>
    Matrix<Item, MatImpl, S>
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

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Dynamic>> Matrix<Item, MatImpl, Dynamic> {
    /// Pad with `ncolumns` columns at the right of the matrix.
    pub fn pad_right(&self, ncolumns: usize) -> MatrixD<Item> {
        let new_dim = (self.shape().0, self.shape().1 + ncolumns);

        let mut new_mat = crate::rlst_mat![Item, new_dim];
        new_mat.subview_mut((0, 0), self.shape()).fill_from(self);
        new_mat
    }

    /// Pad with `ncolumns` columns at the left of the matrix.
    pub fn pad_left(&self, ncolumns: usize) -> MatrixD<Item> {
        let new_dim = (self.shape().0, self.shape().1 + ncolumns);

        let mut new_mat = crate::rlst_mat![Item, new_dim];
        new_mat
            .subview_mut((0, ncolumns), self.shape())
            .fill_from(self);
        new_mat
    }

    /// Pad with `nrows` rows at the top of the matrix.
    pub fn pad_above(&self, nrows: usize) -> MatrixD<Item> {
        let new_dim = (self.shape().0 + nrows, self.shape().1);

        let mut new_mat = crate::rlst_mat![Item, new_dim];
        new_mat
            .subview_mut((nrows, 0), self.shape())
            .fill_from(self);
        new_mat
    }

    /// Pad with `nrows` rows at the bottom of the matrix.
    pub fn pad_below(&self, nrows: usize) -> MatrixD<Item> {
        let new_dim = (self.shape().0 + nrows, self.shape().1);

        let mut new_mat = crate::rlst_mat![Item, new_dim];
        new_mat.subview_mut((0, 0), self.shape()).fill_from(self);
        new_mat
    }

    /// Append `other` to the right of the current matrix.
    pub fn append_right<MatImpl2: MatrixImplTrait<Item, Dynamic>>(
        &self,
        other: &Matrix<Item, MatImpl2, Dynamic>,
    ) -> MatrixD<Item> {
        let mut new_mat = self.pad_right(other.shape().1);
        new_mat
            .subview_mut((0, self.shape().1), other.shape())
            .fill_from(other);
        new_mat
    }

    /// Append `other` to the bottom of the current matrix.
    pub fn append_below<MatImpl2: MatrixImplTrait<Item, Dynamic>>(
        &self,
        other: &Matrix<Item, MatImpl2, Dynamic>,
    ) -> MatrixD<Item> {
        let mut new_mat = self.pad_below(other.shape().0);
        new_mat
            .subview_mut((self.shape().0, 0), other.shape())
            .fill_from(other);
        new_mat
    }
}
