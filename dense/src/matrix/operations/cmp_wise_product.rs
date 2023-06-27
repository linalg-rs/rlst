//! Implementation of the componentwise product.

use crate::op_containers::cmp_wise_product::{CmpWiseProductContainer, CmpWiseProductMat};
use crate::{matrix_ref::MatrixRef, MatrixImplTrait};
use crate::{Matrix, SizeIdentifier};
use rlst_common::traits::CmpWiseProduct;

pub use rlst_common::types::Scalar;

impl<
        'a,
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > CmpWiseProduct<&'a Matrix<Item, MatImpl2, RS, CS>> for Matrix<Item, MatImpl1, RS, CS>
{
    type Out = CmpWiseProductMat<Item, MatImpl1, MatrixRef<'a, Item, MatImpl2, RS, CS>, RS, CS>;

    fn cmp_wise_product(self, other: &'a Matrix<Item, MatImpl2, RS, CS>) -> Self::Out {
        Matrix::new(CmpWiseProductContainer::new(self, other.view()))
    }
}

impl<
        Item: Scalar,
        MatImpl1: MatrixImplTrait<Item, RS, CS>,
        MatImpl2: MatrixImplTrait<Item, RS, CS>,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > CmpWiseProduct<Matrix<Item, MatImpl2, RS, CS>> for Matrix<Item, MatImpl1, RS, CS>
{
    type Out = CmpWiseProductMat<Item, MatImpl1, MatImpl2, RS, CS>;

    fn cmp_wise_product(self, other: Matrix<Item, MatImpl2, RS, CS>) -> Self::Out {
        Matrix::new(CmpWiseProductContainer::new(self, other))
    }
}

#[cfg(test)]
mod test {
    use rlst_common::traits::CmpWiseProduct;

    use crate::rlst_mat;
    use rlst_common::traits::*;

    #[test]
    fn test_cmp_wise_prod() {
        let mut mat1 = rlst_mat![f64, (4, 5)];
        let mut mat2 = rlst_mat![f64, (4, 5)];

        mat1.fill_from_seed_equally_distributed(0);
        mat2.fill_from_seed_equally_distributed(1);

        let res = mat1.view().cmp_wise_product(&mat2).eval();

        assert_eq!(mat1[[2, 3]] * mat2[[2, 3]], res.get_value(2, 3).unwrap());
    }
}
