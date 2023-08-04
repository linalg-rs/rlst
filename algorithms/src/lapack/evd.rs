//! Implement the SVD
use crate::linalg::DenseMatrixLinAlgBuilder;
use crate::traits::evd::*;
use lapacke;
use num::traits::Zero;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::MatrixD;

macro_rules! implement_evd_real {
    ($scalar:ty, $lapack_geev:ident) => {
        impl<'a, Mat: Copy> Evd for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar> + Shape + Stride,
        {
            type T = $scalar;
            fn evd(
                self,
                mode: EigenvectorMode,
            ) -> RlstResult<(
                Vec<<$scalar as Scalar>::Complex>,
                Option<MatrixD<<$scalar as Scalar>::Complex>>,
                Option<MatrixD<<$scalar as Scalar>::Complex>>,
            )> {
                let mut copied = self.into_lapack()?;

                let m = copied.mat.shape().0;
                let n = copied.mat.shape().1;

                if m != n {
                    return Err(RlstError::MatrixNotSquare(m, n));
                }

                let mut wr = vec![<<$scalar as Scalar>::Real as Zero>::zero(); m];
                let mut wi = vec![<<$scalar as Scalar>::Real as Zero>::zero(); m];

                let jobvl;
                let jobvr;

                let mut vl_matrix;
                let mut vr_matrix;
                let vl_data: &mut [Self::T];
                let vr_data: &mut [Self::T];

                // The following two are needed as dummy arrays for the case
                // that the computation of u and v is not requested. Even then
                // we still need to pass a valid reference to the Lapack routine.
                let mut vl_dummy = vec![<Self::T as Zero>::zero(); 1];
                let mut vr_dummy = vec![<Self::T as Zero>::zero(); 1];

                match mode {
                    EigenvectorMode::Compute => {
                        jobvl = b'V';
                        jobvr = b'V';
                        vl_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vr_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vl_data = vl_matrix.as_mut().unwrap().data_mut();
                        vr_data = vr_matrix.as_mut().unwrap().data_mut();
                    }
                    EigenvectorMode::None => {
                        jobvl = b'N';
                        jobvr = b'N';
                        vl_matrix = None;
                        vr_matrix = None;
                        vl_data = vl_dummy.as_mut_slice();
                        vr_data = vr_dummy.as_mut_slice();
                    }
                }

                let info = unsafe {
                    lapacke::$lapack_geev(
                        lapacke::Layout::ColumnMajor,
                        jobvl,
                        jobvr,
                        n as i32,
                        copied.mat.data_mut(),
                        n as i32,
                        wr.as_mut_slice(),
                        wi.as_mut_slice(),
                        vl_data,
                        n as i32,
                        vr_data,
                        n as i32,
                    )
                };

                match info {
                    0 => {
                        let mut vl_complex = None;
                        let mut vr_complex = None;
                        let eigenvalues = wr
                            .iter()
                            .zip(wi.iter())
                            .map(|(&re, &im)| <<$scalar as Scalar>::Complex>::new(re, im))
                            .collect::<Vec<<$scalar as Scalar>::Complex>>();
                        if jobvl == b'V' {
                            vl_complex =
                                Some(rlst_dense::rlst_mat![<$scalar as Scalar>::Complex, (m, m)]);
                            vr_complex =
                                Some(rlst_dense::rlst_mat![<$scalar as Scalar>::Complex, (m, m)]);
                            let mut col = 0;
                            while col < n {
                                if (col < n - 1) && eigenvalues[col] == eigenvalues[1 + col].conj()
                                {
                                    // Case of complex conjugate eigenvalues
                                    for row in 0..m {
                                        let vl_re = vl_matrix
                                            .as_ref()
                                            .unwrap()
                                            .get_value(row, col)
                                            .unwrap();
                                        let vl_im = vl_matrix
                                            .as_ref()
                                            .unwrap()
                                            .get_value(row, 1 + col)
                                            .unwrap();

                                        let vr_re = vr_matrix
                                            .as_ref()
                                            .unwrap()
                                            .get_value(row, col)
                                            .unwrap();
                                        let vr_im = vr_matrix
                                            .as_ref()
                                            .unwrap()
                                            .get_value(row, 1 + col)
                                            .unwrap();

                                        *vl_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(vl_re, vl_im);
                                        *vl_complex
                                            .as_mut()
                                            .unwrap()
                                            .get_mut(row, 1 + col)
                                            .unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(vl_re, -vl_im);

                                        *vr_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(vr_re, vr_im);
                                        *vr_complex
                                            .as_mut()
                                            .unwrap()
                                            .get_mut(row, 1 + col)
                                            .unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(vr_re, -vr_im);
                                    }
                                    col += 2;
                                } else {
                                    // Case of real eigenvalues
                                    for row in 0..m {
                                        *vl_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(
                                                vl_matrix
                                                    .as_ref()
                                                    .unwrap()
                                                    .get_value(row, col)
                                                    .unwrap(),
                                                0.0,
                                            );
                                        *vr_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                            <<$scalar as Scalar>::Complex>::new(
                                                vr_matrix
                                                    .as_ref()
                                                    .unwrap()
                                                    .get_value(row, col)
                                                    .unwrap(),
                                                0.0,
                                            );
                                    }
                                    col += 1;
                                }
                            }
                        }
                        return Ok((eigenvalues, vl_complex, vr_complex));
                    }
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

implement_evd_real!(f64, dgeev);
implement_evd_real!(f32, sgeev);
// implement_svd!(c32, cgesvd);
// implement_svd!(c64, zgesvd);

#[cfg(test)]
mod test {

    use super::*;
    use crate::linalg::LinAlg;
    use rlst_common::tools::PrettyPrint;

    #[test]
    pub fn test_ev() {
        let mut rlst_mat = rlst_dense::rlst_mat![f64, (2, 2)];
        rlst_mat.fill_from_seed_equally_distributed(0);

        let (eigvals, left, right) = rlst_mat.linalg().evd(EigenvectorMode::Compute).unwrap();

        left.unwrap().pretty_print();
        right.unwrap().pretty_print();
    }
}
