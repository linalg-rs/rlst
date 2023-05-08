//! Implementation of Norm2.
use crate::lapack::DenseMatrixLinAlgBuilder;
use rlst_common::traits::{Scalar, Shape, SquareSum};
use rlst_common::types::{RlstError, RlstResult};

use crate::{
    traits::norm2::Norm2,
    traits::svd::{Mode, Svd},
};

impl<'a, T: Scalar, Mat: Shape + SquareSum<T = T>> Norm2 for DenseMatrixLinAlgBuilder<'a, T, Mat>
where
    Self: Svd<T = T>,
{
    type T = T;

    fn norm2(self) -> RlstResult<<Self::T as Scalar>::Real> {
        let shape = self.mat.shape();

        if shape.0 == 0 || shape.1 == 0 {
            return Err(RlstError::MatrixIsEmpty(shape));
        }

        // If we have a vector just use the standard vector norm definition.
        if shape.0 == 1 || shape.1 == 1 {
            return Ok(self.mat.square_sum().sqrt());
        } else {
            // For matrices compute the 2-norm as largest singular value.
            let (s, _, _) = self.svd(Mode::None, Mode::None)?;
            Ok(s[0])
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::linalg::LinAlg;
    use rlst_dense::rlst_rand_mat;

    #[test]
    fn svd_test() {
        let mat = rlst_rand_mat![f64, (4, 3)];
        println!("The norm is {}", mat.linalg().norm2().unwrap());
    }
}
