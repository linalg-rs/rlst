//! Implementation of common matrix traits and methods.

use crate::data_container::{DataContainer, DataContainerMut};
use crate::matrix::Matrix;
use crate::types::Scalar;
use crate::GenericBaseMatrix;
use crate::{traits::*, DefaultLayout};
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
        let shape = result.shape();
        unsafe {
            for col in 0..shape.1 {
                for row in 0..shape.0 {
                    *result.get_unchecked_mut(row, col) = self.get_value_unchecked(row, col);
                }
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
            unsafe { f(self.get1d_unchecked_mut(index)) }
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
        Other: UnsafeRandomAccessByValue<Item = Item> + Shape,
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
                unsafe { *self.get_unchecked_mut(row, col) = other.get_value_unchecked(row, col) };
            }
        }
    }
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = Item>,
        Other: UnsafeRandomAccessByValue<Item = Item> + Shape,
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

#[cfg(test)]
mod test {
    use crate::rlst_rand_mat;
    use rlst_common::tools::ToMatDisplayWrapper_c64;
    use rlst_common::tools::ToMatDisplayWrapper_f64;
    use rlst_common::types::c64;

    // use crate::common_impl::test;

    #[test]
    fn test_display_dense_real_3x1() {
        let mut rlst_vec_real_3x1 = crate::rlst_col_vec![f64, 3];

        rlst_vec_real_3x1[[0, 0]] = 2.3;
        rlst_vec_real_3x1[[1, 0]] = 7.1;
        rlst_vec_real_3x1[[2, 0]] = -143.175;
        println!("{}", rlst_vec_real_3x1.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_real_3x2() {
        let mut rlst_mat_real_3x2 = crate::rlst_mat![f64, (3, 2)];

        rlst_mat_real_3x2[[0, 0]] = 2.3;
        rlst_mat_real_3x2[[0, 1]] = -64.28;
        rlst_mat_real_3x2[[1, 0]] = 7.1;
        rlst_mat_real_3x2[[1, 1]] = -137.16;
        rlst_mat_real_3x2[[2, 0]] = -143.175;
        rlst_mat_real_3x2[[2, 1]] = 8962.7904;
        println!("{}", rlst_mat_real_3x2.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_real_4x4() {
        let mut rlst_mat_real_4x4 = crate::rlst_mat![f64, (4, 4)];

        rlst_mat_real_4x4[[0, 0]] = -42.389645;
        rlst_mat_real_4x4[[0, 1]] = -974917292.0000328;
        rlst_mat_real_4x4[[0, 2]] = 14.0;
        rlst_mat_real_4x4[[0, 3]] = 810.98;
        rlst_mat_real_4x4[[1, 0]] = 78324674.1;
        rlst_mat_real_4x4[[1, 1]] = 1437.769999;
        rlst_mat_real_4x4[[1, 2]] = 3648726.0;
        rlst_mat_real_4x4[[1, 3]] = -19823.64921768;
        rlst_mat_real_4x4[[2, 0]] = -14378721.175;
        rlst_mat_real_4x4[[2, 1]] = 8962.7904;
        rlst_mat_real_4x4[[2, 2]] = -123.456;
        rlst_mat_real_4x4[[2, 3]] = -100_000_000.0;
        rlst_mat_real_4x4[[3, 0]] = 5.3e12;
        rlst_mat_real_4x4[[3, 1]] = -123_456_789.101_1;
        rlst_mat_real_4x4[[3, 2]] = 999_999_999_999.00;
        rlst_mat_real_4x4[[3, 3]] = 0.0;
        println!("{}", rlst_mat_real_4x4.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_complex_3x1() {
        let rlst_vec_complex_3x1 = rlst_rand_mat![c64, (3, 1)];
        println!("{}", rlst_vec_complex_3x1.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_complex_2x2() {
        let rlst_mat_complex_2x2 = rlst_rand_mat![c64, (2, 2)];
        println!("{}", rlst_mat_complex_2x2.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_complex_4x2() {
        let rlst_mat_complex_4x2 = rlst_rand_mat![c64, (4, 2)];
        println!("{}", rlst_mat_complex_4x2.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_complex_2x3() {
        let rlst_mat_complex_2x3 = rlst_rand_mat![c64, (2, 3)];
        println!("{}", rlst_mat_complex_2x3.to_stdout());
        assert!(true);
    }

    #[test]
    fn test_display_dense_complex_4x4() {
        let rlst_mat_complex_4x4 = rlst_rand_mat![c64, (4, 4)];
        println!("{}", rlst_mat_complex_4x4.to_stdout());
        assert!(true);
    }
}
