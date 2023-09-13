//! Implementation of the componentwise product.

use crate::op_containers::cmp_wise_product::{CmpWiseProductContainer, CmpWiseProductMat};
use crate::{matrix_ref::MatrixRef, MatrixImplTrait};
use crate::{Matrix, SizeIdentifier};
use rlst_common::traits::CmpWiseProduct;

pub use rlst_common::types::Scalar;

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > CmpWiseProduct<&'a Matrix<Item, MatImpl2, S>> for Matrix<Item, MatImpl1, S>
{
    type Out = CmpWiseProductMat<Item, MatImpl1, MatrixRef<'a, Item, MatImpl2, S>, S>;

    fn cmp_wise_product(self, other: &'a Matrix<Item, MatImpl2, S>) -> Self::Out {
        Matrix::new(CmpWiseProductContainer::new(self, other.view()))
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, S>,
        MatImpl2: MatrixImplTrait<Item, S>,
        S: SizeIdentifier,
    > CmpWiseProduct<Matrix<Item, MatImpl2, S>> for Matrix<Item, MatImpl1, S>
{
    type Out = CmpWiseProductMat<Item, MatImpl1, MatImpl2, S>;

    fn cmp_wise_product(self, other: Matrix<Item, MatImpl2, S>) -> Self::Out {
        Matrix::new(CmpWiseProductContainer::new(self, other))
    }
}

// pub fn test_simd() {
//     use crate::rlst_rand_mat;
//     use rlst_common::traits::*;

//     let mat1 = rlst_rand_mat![f32, (10, 10)];
//     let mat2 = rlst_rand_mat![f32, (10, 10)];

//     let res = (&mat2 + mat1.view().cmp_wise_product(&mat2)).eval();

//     assert_eq!(res[[0, 0]], mat2[[0, 0]] + mat1[[0, 0]] * mat2[[0, 0]]);
// }

#[cfg(test)]
mod test {
    use rlst_common::traits::CmpWiseProduct;

    use crate::rlst_dynamic_mat;
    use rlst_common::traits::*;

    #[test]
    fn test_cmp_wise_prod() {
        let mut mat1 = rlst_dynamic_mat![f64, (4, 5)];
        let mut mat2 = rlst_dynamic_mat![f64, (4, 5)];

        mat1.fill_from_seed_equally_distributed(0);
        mat2.fill_from_seed_equally_distributed(1);

        let res = mat1.view().cmp_wise_product(&mat2).eval();

        assert_eq!(mat1[[2, 3]] * mat2[[2, 3]], res.get_value(2, 3).unwrap());
    }
}
