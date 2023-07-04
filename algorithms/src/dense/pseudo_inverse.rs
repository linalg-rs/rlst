//! Implementation of Moore-Penrose PseudoInverse
use crate::linalg::DenseMatrixLinAlgBuilder;
use crate::traits::pseudo_inverse::Pinv;
use crate::traits::svd::Mode;
use crate::traits::svd::Svd;

use num::Float;
use rlst_common::traits::*;
use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::MatrixD;

impl<'a, T: Scalar + Float, Mat: Shape> Pinv for DenseMatrixLinAlgBuilder<'a, T, Mat>
where
    Self: Svd<T = T>,
{
    type T = T;

    fn pinv(
        self,
    ) -> RlstResult<(
        Vec<<Self::T as Scalar>::Real>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )> {
        let shape = self.mat.shape();

        if shape.0 == 0 || shape.1 == 0 {
            return Err(RlstError::MatrixIsEmpty(shape));
        }

        // If we have a vector return error
        if shape.0 == 1 || shape.1 == 1 {
            Err(RlstError::SingleDimensionError {
                expected: 2,
                actual: 1,
            })
        } else {
            // For matrices compute the full SVD
            let (mut s, u, vt) = self.svd(Mode::All, Mode::All)?;
            let u = u.unwrap();
            let vt = vt.unwrap();

            // Compute a threshold based on the maximum singular value
            let max_s = s[0];
            let threshold = T::real(4.).mul_real(T::real(T::epsilon())).mul_real(max_s);

            // Filter singular values below this threshold
            for s in s.iter_mut() {
                if *s > threshold {
                    *s = T::real(1.0) / *s;
                } else {
                    *s = T::real(0.)
                }
            }

            // Return pseudo-inverse in component form
            let v = Some(vt.transpose().eval());
            let ut = Some(u.transpose().eval());

            Ok((s, ut, v))
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::linalg::LinAlg;
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rlst_common::tools::PrettyPrint;
    use rlst_dense::{rlst_mat, rlst_rand_mat, Dot};

    #[test]
    fn test_pinv() {
        let dim: usize = 5;
        let mut mat = rlst_rand_mat![f64, (dim, dim)];

        let (s, ut, v) = mat.linalg().pinv().unwrap();

        let ut = ut.unwrap();
        let v = v.unwrap();

        let mut mat_s = rlst_mat![f64, (s.len(), s.len())];
        for i in 0..s.len() {
            mat_s[[i, i]] = s[i];
        }

        let mut inv = v.dot(&mat_s).dot(&ut);

        let actual = inv.dot(&mat);

        // Expect the identity matrix
        let mut expected = actual.new_like_self();
        for i in 0..dim {
            expected[[i, i]] = 1.0
        }

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }
}
