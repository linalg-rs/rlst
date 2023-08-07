//! Implementation of common matrix traits and methods.

use crate::matrix::Matrix;
use crate::types::Scalar;
use crate::RefMatMut;
use crate::{traits::*, DefaultLayout};
use crate::{MatrixD, RefMat};
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
            unsafe { f(self.get1d_unchecked_mut(index)) }
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
        Other: UnsafeRandomAccessByValue<Item = Item> + Shape,
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

        for col in 0..self.shape().1 {
            for row in 0..self.shape().0 {
                unsafe { *self.get_unchecked_mut(row, col) = other.get_value_unchecked(row, col) };
            }
        }
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Other: UnsafeRandomAccessByValue<Item = Item> + Shape,
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

        for col in 0..self.shape().1 {
            for row in 0..self.shape().0 {
                unsafe {
                    *self.get_unchecked_mut(row, col) += alpha * other.get_value_unchecked(row, col)
                };
            }
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
        let shape = self.shape();

        let mut result = <<Self::T as Scalar>::Real as Zero>::zero();
        for col in 0..shape.1 {
            for row in 0..shape.0 {
                let value = unsafe { self.get_value_unchecked(row, col) };
                result += value.square();
            }
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
    pub fn view(&self) -> RefMat<Item, MatImpl, RS, CS> {
        Matrix::from_ref(self)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTraitMut<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, RS, CS>
{
    pub fn view_mut(&mut self) -> RefMatMut<Item, MatImpl, RS, CS> {
        Matrix::from_ref_mut(self)
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
