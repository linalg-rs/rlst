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

#[cfg(test)]
mod test {

    use super::*;
    use crate::{rlst_fixed_rand_mat, rlst_rand_mat};
    use rlst_common::types::c64;

    #[test]
    fn test_conj_transpose() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let transpose_mat = mat.conj_transpose();

        assert_eq!(transpose_mat.shape(), (4, 3));

        for col_index in 0..mat.shape().1 {
            for row_index in 0..mat.shape().0 {
                assert_eq!(
                    mat[[row_index, col_index]],
                    transpose_mat[[col_index, row_index]].conj()
                );
            }
        }
    }

    #[test]
    fn test_transpose() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let transpose_mat = mat.transpose();

        assert_eq!(transpose_mat.shape(), (4, 3));

        for col_index in 0..mat.shape().1 {
            for row_index in 0..mat.shape().0 {
                assert_eq!(
                    mat[[row_index, col_index]],
                    transpose_mat[[col_index, row_index]]
                );
            }
        }
    }

    #[test]
    fn test_fixed_mat_transpose() {
        let mat = rlst_fixed_rand_mat![c64, 2, 3];

        let transpose_mat = mat.transpose();

        assert_eq!(transpose_mat.shape(), (3, 2));

        for col_index in 0..mat.shape().1 {
            for row_index in 0..mat.shape().0 {
                assert_eq!(
                    mat[[row_index, col_index]],
                    transpose_mat[[col_index, row_index]]
                );
            }
        }
    }
}
