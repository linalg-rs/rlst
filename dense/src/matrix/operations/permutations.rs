use crate::MatrixImplTrait;
pub use crate::{DataContainer, DefaultLayout, Matrix, SizeIdentifier};
pub use rlst_common::traits::Transpose;
use rlst_common::traits::{
    NewLikeSelf, PermuteColumns, PermuteRows, RandomAccessByValue, RandomAccessMut, Shape,
};
pub use rlst_common::types::Scalar;

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > PermuteColumns for Matrix<Item, MatImpl, RS, CS>
where
    Self: NewLikeSelf + RandomAccessByValue<Item = Item>,
    <Self as NewLikeSelf>::Out: RandomAccessMut<Item = Item>,
{
    type Out = <Self as NewLikeSelf>::Out;
    fn permute_columns(&self, permutation: &[usize]) -> Self::Out {
        let mut new_mat = self.new_like_self();

        let shape = self.shape();

        assert_eq!(
            permutation.len(),
            shape.1,
            "Length of permutation vector is {}. But expected {}.",
            permutation.len(),
            shape.1
        );

        for (col_index, &perm_value) in permutation.iter().enumerate() {
            for row_index in 0..shape.0 {
                *new_mat.get_mut(row_index, col_index).unwrap() =
                    self.get_value(row_index, perm_value).unwrap();
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
    > PermuteRows for Matrix<Item, MatImpl, RS, CS>
where
    Self: NewLikeSelf + RandomAccessByValue<Item = Item>,
    <Self as NewLikeSelf>::Out: RandomAccessMut<Item = Item>,
{
    type Out = <Self as NewLikeSelf>::Out;
    fn permute_rows(&self, permutation: &[usize]) -> Self::Out {
        let mut new_mat = self.new_like_self();

        let shape = self.shape();

        assert_eq!(
            permutation.len(),
            shape.0,
            "Length of permutation vector is {}. But expected {}.",
            permutation.len(),
            shape.0
        );

        for col_index in 0..shape.1 {
            for (row_index, &perm_value) in permutation.iter().enumerate() {
                *new_mat.get_mut(row_index, col_index).unwrap() =
                    self.get_value(perm_value, col_index).unwrap();
            }
        }
        new_mat
    }
}
