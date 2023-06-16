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
            Ok(self.mat.square_sum().sqrt())
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
    use approx::assert_ulps_eq;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use rlst_dense::rlst_col_vec;
    use rlst_dense::rlst_mat;

    #[test]
    fn test_vector_norm() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut mat = rlst_col_vec![f64, 2];
        mat.fill_from_equally_distributed(&mut rng);

        let expected = (mat[[0, 0]] * mat[[0, 0]] + mat[[1, 0]] * mat[[1, 0]]).sqrt();
        let actual = mat.linalg().norm2().unwrap();

        assert_ulps_eq![expected, actual, max_ulps = 10];
    }

    #[test]
    fn test_matrix_norm() {
        let mut mat = rlst_mat![f64, (2, 2)];

        mat[[0, 0]] = -1.0;
        mat[[1, 1]] = 0.5;

        assert_ulps_eq![mat.linalg().norm2().unwrap(), 1.0, max_ulps = 10];
    }
}
