//! Implementation of matrix transpose and conjugate transpose

use crate::MatrixImplTrait;
pub use crate::{DataContainer, DefaultLayout, Matrix, SizeIdentifier};
pub use rlst_common::traits::Transpose;
use rlst_common::traits::{
    ConjTranspose, NewLikeTranspose, RandomAccessByValue, RandomAccessMut, Shape,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
pub use rlst_common::types::Scalar;

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Transpose for Matrix<Item, MatImpl, RS, CS>
where
    Self: NewLikeTranspose + RandomAccessByValue<Item = Item>,
    <Self as NewLikeTranspose>::Out: RandomAccessMut<Item = Item>,
{
    type Out = <Self as NewLikeTranspose>::Out;
    fn transpose(&self) -> Self::Out {
        let mut new_mat = self.new_like_transpose();

        let shape = self.shape();

        for col_index in 0..shape.1 {
            for row_index in 0..shape.0 {
                unsafe {
                    *new_mat.get_unchecked_mut(col_index, row_index) =
                        self.get_value_unchecked(row_index, col_index);
                };
            }
        }
        new_mat
    }
}

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > ConjTranspose for Matrix<Item, MatImpl, RS, CS>
where
    Self: NewLikeTranspose + RandomAccessByValue<Item = Item>,
    <Self as NewLikeTranspose>::Out: RandomAccessMut<Item = Item>,
{
    type Out = <Self as NewLikeTranspose>::Out;
    fn conj_transpose(&self) -> Self::Out {
        let mut new_mat = self.new_like_transpose();

        let shape = self.shape();

        for col_index in 0..shape.1 {
            for row_index in 0..shape.0 {
                unsafe {
                    *new_mat.get_unchecked_mut(col_index, row_index) =
                        self.get_value_unchecked(row_index, col_index).conj();
                };
            }
        }
        new_mat
    }
}
