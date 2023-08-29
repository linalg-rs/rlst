//! Implementation of matrix transpose

use crate::op_containers::transpose::{TransposeContainer, TransposeMat};
use crate::MatrixImplTrait;
use crate::{Matrix, SizeIdentifier};
use rlst_common::traits::Transpose;

pub use rlst_common::types::Scalar;

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier> Transpose
    for Matrix<Item, MatImpl, S>
{
    type Out = TransposeMat<Item, MatImpl, S>;

    fn transpose(self) -> Self::Out {
        Matrix::new(TransposeContainer::new(self))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::rlst_rand_mat;
    use rlst_common::traits::*;
    use rlst_common::types::c64;

    #[test]
    fn test_transpose() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let transpose_mat = mat.view().transpose().eval();

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

    // #[test]
    // fn test_fixed_mat_transpose() {
    //     let mat = rlst_fixed_rand_mat![c64, 2, 3];

    //     let transpose_mat = mat.transpose().eval();

    //     assert_eq!(transpose_mat.shape(), (3, 2));

    //     for col_index in 0..mat.shape().1 {
    //         for row_index in 0..mat.shape().0 {
    //             assert_eq!(
    //                 mat[[row_index, col_index]],
    //                 transpose_mat[[col_index, row_index]]
    //             );
    //         }
    //     }
    // }
}
