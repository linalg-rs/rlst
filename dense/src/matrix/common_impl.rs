//! Implementation of common matrix traits and methods.

use crate::data_container::{DataContainer, DataContainerMut};
use crate::matrix::Matrix;
use crate::types::Scalar;
use crate::RefMat;
use crate::{traits::*, DefaultLayout};
use crate::{GenericBaseMatrix, RefMatMut};
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
    <Self as NewLikeSelf>::Out: Shape + RandomAccessMut<Item = Item>,
{
    type Out = <Self as NewLikeSelf>::Out;

    fn eval(&self) -> Self::Out {
        let mut result = self.new_like_self();
        for index in 0..self.layout().number_of_elements() {
            *result.get1d_mut(index).unwrap() = self.get1d_value(index).unwrap();
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

impl<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainerMut<Item = Item>>
    ForEach for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;
    fn for_each<F: FnMut(&mut Self::T)>(&mut self, mut f: F) {
        for index in 0..self.layout().number_of_elements() {
            f(self.get1d_mut(index).unwrap())
        }
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    RawAccess for GenericBaseMatrix<Item, Data, RS, CS>
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
        self.0.get_slice(0, self.layout().number_of_elements())
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    RawAccessMut for GenericBaseMatrix<Item, Data, RS, CS>
{
    fn get_pointer_mut(&mut self) -> *mut Item {
        self.0.get_pointer_mut()
    }

    fn get_slice_mut(&mut self, first: usize, last: usize) -> &mut [Item] {
        self.0.get_slice_mut(first, last)
    }

    fn data_mut(&mut self) -> &mut [Item] {
        self.0.get_slice_mut(0, self.layout().number_of_elements())
    }
}

impl<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainerMut<Item = Item>>
    ScaleInPlace for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;

    fn scale_in_place(&mut self, alpha: Self::T) {
        self.for_each(|elem| *elem = alpha * *elem);
    }
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = Item>,
        Other: RandomAccessByValue<Item = Item> + Shape,
    > FillFrom<Other> for GenericBaseMatrix<Item, Data, RS, CS>
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
                *self.get_mut(row, col).unwrap() = other.get_value(row, col).unwrap();
            }
        }
    }
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = Item>,
        Other: RandomAccessByValue<Item = Item> + Shape,
    > SumInto<Other> for GenericBaseMatrix<Item, Data, RS, CS>
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

        for elem in 0..self.number_of_elements() {
            *self.get1d_mut(elem).unwrap() += alpha * other.get1d_value(elem).unwrap();
        }
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

pub fn test_simd() {
    let mat1 = crate::rlst_mat![f32, (20, 20)];
    let mat2 = crate::rlst_mat![f32, (20, 20)];

    let res = (mat1.view() + mat2.view()).eval();
    println!("Res {}", res[[0, 0]]);
}
