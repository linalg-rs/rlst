//! Implementation of common matrix traits and methods.

use crate::matrix::{Matrix, MatrixD};
use crate::types::Scalar;
use crate::{traits::*, DefaultLayout};
use rlst_common::traits::properties::*;

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
    > Matrix<Item, MatImpl, RS, CS>
{
    /// Evaluate into a new matrix.
    pub fn eval(self) -> MatrixD<Item> {
        let dim = self.layout().dim();
        let mut result = crate::rlst_mat![Item, dim];
        unsafe {
            for row in 0..dim.0 {
                for col in 0..dim.1 {
                    *result.get_unchecked_mut(row, col) = self.get_value_unchecked(row, col);
                }
            }
        }
        result
    }
}

impl<T: Scalar> Clone for MatrixD<T> {
    fn clone(&self) -> Self {
        let mut out = crate::rlst_mat!(T, self.shape());

        for col in 0..self.shape().1 {
            for row in 0..self.shape().0 {
                unsafe { *out.get_unchecked_mut(row, col) = *self.get_unchecked(row, col) }
            }
        }
        out
    }
}

// macro_rules! eval_dynamic_matrix {
//     ($L:ident) => {
//         impl<Item: Scalar, MatImpl: MatrixTrait<Item, $L, Dynamic, Dynamic>>
//             Matrix<Item, MatImpl, $L, Dynamic, Dynamic>
//         {
//             pub fn eval(&self) -> MatrixD<Item, $L> {
//                 let dim = self.dim();
//                 let mut result = MatrixD::<Item, $L>::from_zeros(dim.0, dim.1);
//                 for index in 0..self.number_of_elements() {
//                     unsafe { *result.get1d_unchecked_mut(index) = self.get1d_unchecked(index) };
//                 }
//                 result
//             }
//         }
//     };
// }

// macro_rules! eval_fixed_matrix {
//     ($L:ident, $RS:ty, $CS:ty) => {
//         impl<Item: Scalar, MatImpl: MatrixTrait<Item, $L, $RS, $CS>>
//             Matrix<Item, MatImpl, $L, $RS, $CS>
//         {
//             pub fn eval(
//                 &self,
//             ) -> Matrix<
//                 Item,
//                 BaseMatrix<Item, ArrayContainer<Item, { <$RS>::N * <$CS>::N }>, $L, $RS, $CS>,
//                 $L,
//                 $RS,
//                 $CS,
//             > {
//                 let mut result = Matrix::<
//                     Item,
//                     BaseMatrix<Item, ArrayContainer<Item, { <$RS>::N * <$CS>::N }>, $L, $RS, $CS>,
//                     $L,
//                     $RS,
//                     $CS,
//                 >::from_zeros();
//                 for index in 0..self.number_of_elements() {
//                     unsafe { *result.get1d_unchecked_mut(index) = self.get1d_unchecked(index) };
//                 }
//                 result
//             }
//         }
//     };
// }

// eval_dynamic_matrix!(CLayout);
// eval_dynamic_matrix!(FLayout);

// eval_fixed_matrix!(CLayout, Fixed2, Fixed2);
// eval_fixed_matrix!(CLayout, Fixed3, Fixed2);
// eval_fixed_matrix!(CLayout, Fixed2, Fixed3);
// eval_fixed_matrix!(CLayout, Fixed3, Fixed3);

// eval_fixed_matrix!(FLayout, Fixed2, Fixed2);
// eval_fixed_matrix!(FLayout, Fixed3, Fixed2);
// eval_fixed_matrix!(FLayout, Fixed2, Fixed3);
// eval_fixed_matrix!(FLayout, Fixed3, Fixed3);
