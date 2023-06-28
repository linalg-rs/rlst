//! Implementation of matrix conjugate

use crate::op_containers::conjugate::{ConjugateContainer, ConjugateMat};
use crate::MatrixImplTrait;
use crate::{Matrix, SizeIdentifier};
use rlst_common::traits::Conjugate;

pub use rlst_common::types::Scalar;

impl<
        Item: Scalar,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > Conjugate for Matrix<Item, MatImpl, RS, CS>
{
    type Out = ConjugateMat<Item, MatImpl, RS, CS>;

    fn conj(self) -> Self::Out {
        Matrix::new(ConjugateContainer::new(self))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::rlst_rand_mat;
    use rlst_common::traits::*;
    use rlst_common::types::c64;

    #[test]
    fn test_conj_transpose() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let conj_transpose_mat = mat.view().conj().transpose();

        assert_eq!(conj_transpose_mat.shape(), (4, 3));

        for col_index in 0..mat.shape().1 {
            for row_index in 0..mat.shape().0 {
                assert_eq!(
                    mat[[row_index, col_index]],
                    conj_transpose_mat
                        .get_value(col_index, row_index)
                        .unwrap()
                        .conj()
                );
            }
        }
    }

    #[test]
    fn test_conjugate() {
        let mat = rlst_rand_mat![c64, (3, 4)];

        let conjugate_mat = mat.view().transpose().eval();

        assert_eq!(conjugate_mat.shape(), (4, 3));

        for col_index in 0..mat.shape().1 {
            for row_index in 0..mat.shape().0 {
                assert_eq!(
                    mat[[row_index, col_index]],
                    conjugate_mat[[col_index, row_index]]
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
