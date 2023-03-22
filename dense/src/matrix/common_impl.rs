//! Implementation of common matrix traits and methods.

use crate::matrix::{Matrix, MatrixD};
use crate::traits::*;
use crate::types::{IndexType, Scalar};

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Layout for Matrix<Item, MatImpl, L, RS, CS>
{
    type Impl = L;
    fn layout(&self) -> &Self::Impl {
        self.0.layout()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > SizeType for Matrix<Item, MatImpl, L, RS, CS>
{
    type R = RS;
    type C = CS;
}

impl<
        Item: Scalar,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByValue for Matrix<Item, MatImpl, L, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: IndexType, col: IndexType) -> Self::Item {
        self.0.get_value_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: IndexType) -> Self::Item {
        self.0.get1d_value_unchecked(index)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTraitMut<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessMut for Matrix<Item, MatImpl, L, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::Item {
        self.0.get_unchecked_mut(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item {
        self.0.get1d_unchecked_mut(index)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTraitAccessByRef<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > UnsafeRandomAccessByRef for Matrix<Item, MatImpl, L, RS, CS>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &Self::Item {
        self.0.get_unchecked(row, col)
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: IndexType) -> &Self::Item {
        self.0.get1d_unchecked(index)
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTraitAccessByRef<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::Index<[IndexType; 2]> for Matrix<Item, MatImpl, L, RS, CS>
{
    type Output = Item;

    fn index(&self, index: [IndexType; 2]) -> &Self::Output {
        self.get(index[0], index[1]).unwrap()
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixTraitMut<Item, L, RS, CS> + MatrixTraitAccessByRef<Item, L, RS, CS>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > std::ops::IndexMut<[IndexType; 2]> for Matrix<Item, MatImpl, L, RS, CS>
{
    fn index_mut(&mut self, index: [IndexType; 2]) -> &mut Self::Output {
        self.get_mut(index[0], index[1]).unwrap()
    }
}

impl<Item: Scalar, L: LayoutType, MatImpl: MatrixTrait<Item, L, Dynamic, Dynamic>>
    Matrix<Item, MatImpl, L, Dynamic, Dynamic>
{
    /// Evaluate into a new matrix.
    pub fn eval(&self) -> MatrixD<Item, <L as LayoutType>::IndexLayout> {
        let dim = self.layout().dim();
        let mut result =
            MatrixD::<Item, <L as LayoutType>::IndexLayout>::zeros_from_dim(dim.0, dim.1);
        unsafe {
            for index in 0..self.layout().number_of_elements() {
                *result.get1d_unchecked_mut(index) = self.get1d_value_unchecked(index);
            }
        }
        result
    }
}

impl<
        Item: Scalar,
        L: LayoutType,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Matrix<Item, MatImpl, L, RS, CS>
{
    /// Return dimension of the matrix.
    pub fn dim(&self) -> (IndexType, IndexType) {
        self.layout().dim()
    }
}

impl<Item: Scalar, L: LayoutType, MatImpl: MatrixTrait<Item, L, Fixed1, Dynamic>>
    Matrix<Item, MatImpl, L, Fixed1, Dynamic>
{
    /// Return length of a vector.
    pub fn length(&self) -> IndexType {
        self.layout().dim().1
    }
}

impl<Item: Scalar, L: LayoutType, MatImpl: MatrixTrait<Item, L, Dynamic, Fixed1>>
    Matrix<Item, MatImpl, L, Dynamic, Fixed1>
{
    /// Return length of a vector.
    pub fn length(&self) -> IndexType {
        self.layout().dim().0
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
