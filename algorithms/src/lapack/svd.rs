//! Implement the SVD
use crate::linalg::DenseMatrixLinAlgBuilder;
use crate::traits::svd::Mode;
use crate::traits::svd::Svd;
use lapacke;
use num::traits::Zero;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_mat, MatrixD};

macro_rules! implement_svd {
    ($scalar:ty, $lapack_gesvd:ident) => {
        impl<'a, Mat: Copy> Svd for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar> + Shape + Stride,
        {
            type T = $scalar;
            fn svd(
                self,
                u_mode: Mode,
                vt_mode: Mode,
            ) -> RlstResult<(
                Vec<<$scalar as Scalar>::Real>,
                Option<MatrixD<$scalar>>,
                Option<MatrixD<$scalar>>,
            )> {
                let mut copied = self.into_lapack()?;
                let m = copied.mat.shape().0 as i32;
                let n = copied.mat.shape().1 as i32;
                let k = std::cmp::min(m, n);
                let lda = copied.mat.stride().1 as i32;

                let mut s_values = vec![<<$scalar as Scalar>::Real as Zero>::zero(); k as usize];
                let mut superb =
                    vec![<<$scalar as Scalar>::Real as Zero>::zero(); (k - 1) as usize];
                let jobu;
                let jobvt;
                let mut u_matrix;
                let mut vt_matrix;
                let ldu;
                let ldvt;
                let u_data: &mut [Self::T];
                let vt_data: &mut [Self::T];

                // The following two are needed as dummy arrays for the case
                // that the computation of u and v is not requested. Even then
                // we still need to pass a valid reference to the Lapack routine.
                let mut u_dummy = vec![<Self::T as Zero>::zero(); 1];
                let mut vt_dummy = vec![<Self::T as Zero>::zero(); 1];

                match u_mode {
                    Mode::All => {
                        jobu = b'A';
                        u_matrix = Some(rlst_mat![$scalar, (m as usize, m as usize)]);
                        u_data = u_matrix.as_mut().unwrap().data_mut();
                        ldu = m as i32;
                    }
                    Mode::Slim => {
                        jobu = b'S';
                        u_matrix = Some(rlst_mat![$scalar, (m as usize, k as usize)]);
                        u_data = u_matrix.as_mut().unwrap().data_mut();
                        ldu = m as i32;
                    }
                    Mode::None => {
                        jobu = b'N';
                        u_matrix = None;
                        u_data = u_dummy.as_mut_slice();
                        ldu = m as i32;
                    }
                };

                match vt_mode {
                    Mode::All => {
                        jobvt = b'A';
                        vt_matrix = Some(rlst_mat![$scalar, (n as usize, n as usize)]);
                        vt_data = vt_matrix.as_mut().unwrap().data_mut();
                        ldvt = n as i32;
                    }
                    Mode::Slim => {
                        jobvt = b'S';
                        vt_matrix = Some(rlst_mat![$scalar, (k as usize, n as usize)]);
                        vt_data = vt_matrix.as_mut().unwrap().data_mut();
                        ldvt = k as i32;
                    }
                    Mode::None => {
                        jobvt = b'N';
                        vt_matrix = None;
                        vt_data = vt_dummy.as_mut_slice();
                        ldvt = k as i32;
                    }
                }

                let info = unsafe {
                    lapacke::$lapack_gesvd(
                        lapacke::Layout::ColumnMajor,
                        jobu,
                        jobvt,
                        m,
                        n,
                        copied.mat.data_mut(),
                        lda,
                        s_values.as_mut_slice(),
                        u_data,
                        ldu,
                        vt_data,
                        ldvt,
                        superb.as_mut_slice(),
                    )
                };

                match info {
                    0 => return Ok((s_values, u_matrix, vt_matrix)),
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

implement_svd!(f64, dgesvd);
implement_svd!(f32, sgesvd);
implement_svd!(c32, cgesvd);
implement_svd!(c64, zgesvd);

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use rand::SeedableRng;

    use super::*;
    use crate::linalg::LinAlg;
    use rand_chacha::ChaCha8Rng;
    use rlst_dense::Dot;

    #[test]
    fn test_thick_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 3;
        let n = 4;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) =
            mat.linalg().svd(Mode::Slim, Mode::Slim).unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }

    #[test]
    fn test_thin_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 4;
        let n = 3;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) =
            mat.linalg().svd(Mode::Slim, Mode::Slim).unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }

    #[test]
    fn test_full_svd() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let m = 5;
        let n = 5;
        let k = std::cmp::min(m, n);
        let mut mat = rlst_mat!(f64, (m, n));

        mat.fill_from_equally_distributed(&mut rng);
        let expected = mat.copy();

        let (singular_values, u_matrix, vt_matrix) =
            mat.linalg().svd(Mode::All, Mode::All).unwrap();

        let u_matrix = u_matrix.unwrap();
        let vt_matrix = vt_matrix.unwrap();

        let mut sigma = rlst_mat!(f64, (k, k));
        for index in 0..k {
            sigma[[index, index]] = singular_values[index];
        }

        let tmp = sigma.dot(&vt_matrix);

        let actual = u_matrix.dot(&tmp);

        for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
            assert_relative_eq!(a, e, epsilon = 1E-13);
        }
    }
}
