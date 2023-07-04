//! Implementation of Moore-Penrose PseudoInverse
use crate::linalg::DenseMatrixLinAlgBuilder;
use crate::traits::pseudo_inverse::Pinv;
use crate::traits::svd::Mode;
use crate::traits::svd::Svd;

use num::Float;
use rlst_common::traits::*;
use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::rlst_pointer_mat;
use rlst_dense::Dynamic;
use rlst_dense::LayoutType;
use rlst_dense::MatrixD;
use rlst_dense::{RawAccess, Stride};

impl<'a, T: Scalar + Float, Mat: Shape + Copy> Pinv for DenseMatrixLinAlgBuilder<'a, T, Mat>
where
    Self: Svd<T = T>,
    <Mat as Copy>::Out: RawAccess<T = T> + Shape + Stride,
{
    type T = T;

    fn pinv(
        self,
        threshold: Option<Self::T>,
    ) -> RlstResult<(
        Option<Vec<<Self::T as Scalar>::Real>>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )> {
        let shape = self.mat.shape();

        if shape.0 == 0 || shape.1 == 0 {
            return Err(RlstError::MatrixIsEmpty(shape));
        }

        if shape.0 == 1 || shape.1 == 1 {
            // Find innner product of vector with itself, used as a scaling
            let copied = self.mat.copy();
            let shape = copied.shape();
            let stride = copied.stride();
            let data = copied.data().as_ptr();
            let vec = unsafe { rlst_pointer_mat!['a, T, data, shape, stride] }.eval();

            let mut inner = T::real(0.);
            for i in 0..shape.0 {
                inner += T::real(vec.data()[i] * vec.data()[i]);
            }

            let t;
            if let Some(threshold) = threshold {
                t = T::real(threshold)
            } else {
                t = T::real(T::epsilon());
            }

            if inner > t {
                // Find transpose
                let scaling = Some(vec![T::real(1.) / inner]);
                let transpose = vec.copy().transpose().eval();

                Ok((scaling, Some(transpose), None))
            } else {
                // Not defined in case where inner product is zero
                Err(RlstError::OperationFailed(
                    "Pseudo-Inverse Not Defined For This Vector".to_string(),
                ))
            }
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
            let s = Some(s);
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
    use rlst_dense::{rlst_mat, rlst_rand_col_vec, rlst_rand_mat, Dot};

    #[test]
    fn test_pinv_matrix() {
        let dim: usize = 5;
        let mut mat = rlst_rand_mat![f64, (dim, dim)];

        let (s, ut, v) = mat.linalg().pinv(None).unwrap();

        let ut = ut.unwrap();
        let v = v.unwrap();
        let s = s.unwrap();

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

    #[test]
    fn test_pinv_vector() {
        let dim: usize = 5;
        let mut col_vec = rlst_rand_col_vec![f64, dim];

        let (scaling, transpose, _) = col_vec.linalg().pinv(None).unwrap();

        let scaling = scaling.unwrap();
        let transpose = transpose.unwrap();

        let inv = (transpose * scaling[0]).eval();
        let actual = inv.dot(&col_vec);

        let expected = 1.0;

        assert_relative_eq!(actual[[0, 0]], expected, epsilon = 1E-13);
    }
}
