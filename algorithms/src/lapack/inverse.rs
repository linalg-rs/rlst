//! Implementation of the matrix inverse.

use crate::linalg::DenseMatrixLinAlgBuilder;
use lapacke;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult};
use rlst_dense::MatrixD;

use crate::traits::inverse::Inverse;

macro_rules! implement_inverse {
    ($scalar:ty, $lapack_getrf:ident, $lapack_getri:ident) => {
        impl Inverse for DenseMatrixLinAlgBuilder<$scalar> {
            type Out = MatrixD<$scalar>;

            fn inverse(self) -> RlstResult<Self::Out> {
                let mut mat = self.mat;

                if mat.shape().0 != mat.shape().1 {
                    return Err(RlstError::MatrixNotSquare(mat.shape().0, mat.shape().1));
                }

                let n = mat.shape().0;
                let mut ipiv: Vec<i32> = vec![0; n];
                let info = unsafe {
                    lapacke::$lapack_getrf(
                        lapacke::Layout::ColumnMajor,
                        n as i32,
                        n as i32,
                        mat.data_mut(),
                        n as i32,
                        ipiv.as_mut_slice(),
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let info = unsafe {
                    lapacke::$lapack_getri(
                        lapacke::Layout::ColumnMajor,
                        n as i32,
                        mat.data_mut(),
                        n as i32,
                        ipiv.as_slice(),
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }
                Ok(mat)
            }
        }
    };
}

implement_inverse!(f64, dgetrf, dgetri);
implement_inverse!(f32, sgetrf, sgetri);
implement_inverse!(c32, cgetrf, cgetri);
implement_inverse!(c64, zgetrf, zgetri);

#[cfg(test)]
mod test {
    use crate::linalg::LinAlg;

    use super::*;
    use rlst_common::assert_matrix_abs_diff_eq;
    use rlst_dense::Dot;

    use paste::paste;

    macro_rules! test_impl {
        ($scalar:ty, $tol:expr) => {
            paste! {
                    #[test]
                    fn [<test_inverse_$scalar>]() {
                        let mut rlst_mat = rlst_dense::rlst_dynamic_mat![$scalar, (2, 2)];

                        rlst_mat.fill_from_seed_equally_distributed(0);

                        let inverse = rlst_mat.linalg().inverse().unwrap();

                        let ident_actual = inverse.dot(&rlst_mat);
                        let ident_expected = MatrixD::<$scalar>::identity((2, 2));

                        assert_matrix_abs_diff_eq!(ident_actual, ident_expected, $tol);
                    }
            }
        };
    }

    test_impl!(f32, 1E-5);
    test_impl!(f64, 1E-12);
    test_impl!(c32, 1E-5);
    test_impl!(c64, 1E-12);
}
