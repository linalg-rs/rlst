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
                fn convert_eigvecs_to_complex(
                    n: usize,
                    eigenvalues: &[<$scalar as Scalar>::Complex],
                    ev_real: &Option<MatrixD<$scalar>>,
                    ev_complex: &mut Option<MatrixD<<$scalar as Scalar>::Complex>>,
                ) {
                    let mut col = 0;
                    while col < n {
                        if (col < n - 1) && eigenvalues[col] == eigenvalues[1 + col].conj() {
                            // Case of complex conjugate eigenvalues
                            for row in 0..n {
                                let re = ev_real.as_ref().unwrap().get_value(row, col).unwrap();
                                let im = ev_real.as_ref().unwrap().get_value(row, 1 + col).unwrap();

                                *ev_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                    <<$scalar as Scalar>::Complex>::new(re, im);
                                *ev_complex.as_mut().unwrap().get_mut(row, 1 + col).unwrap() =
                                    <<$scalar as Scalar>::Complex>::new(re, -im);
                            }
                            col += 2;
                        } else {
                            // Case of real eigenvalues
                            for row in 0..n {
                                *ev_complex.as_mut().unwrap().get_mut(row, col).unwrap() =
                                    <<$scalar as Scalar>::Complex>::new(
                                        ev_real.as_ref().unwrap().get_value(row, col).unwrap(),
                                        0.0,
                                    );
                            }
                            col += 1;
                        }
                    }
                }

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
                    EigenvectorMode::All => {
                        jobvr = b'V';
                        vr_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vr_data = vr_matrix.as_mut().unwrap().data_mut();
                        jobvl = b'V';
                        vl_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vl_data = vl_matrix.as_mut().unwrap().data_mut();
                    }
                    EigenvectorMode::Right => {
                        jobvr = b'V';
                        vr_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vr_data = vr_matrix.as_mut().unwrap().data_mut();
                        jobvl = b'N';
                        vl_matrix = None;
                        vl_data = vl_dummy.as_mut_slice();
                    }
                    EigenvectorMode::Left => {
                        jobvl = b'V';
                        vl_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vl_data = vl_matrix.as_mut().unwrap().data_mut();
                        jobvr = b'N';
                        vr_matrix = None;
                        vr_data = vr_dummy.as_mut_slice();
                    }
                    EigenvectorMode::None => {
                        jobvr = b'N';
                        vr_matrix = None;
                        vr_data = vr_dummy.as_mut_slice();
                        jobvl = b'N';
                        vl_matrix = None;
                        vl_data = vl_dummy.as_mut_slice();
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
                            convert_eigvecs_to_complex(
                                n,
                                &eigenvalues,
                                &vl_matrix,
                                &mut vl_complex,
                            );
                        }
                        if jobvr == b'V' {
                            vr_complex =
                                Some(rlst_dense::rlst_mat![<$scalar as Scalar>::Complex, (m, m)]);
                            convert_eigvecs_to_complex(
                                n,
                                &eigenvalues,
                                &vr_matrix,
                                &mut vr_complex,
                            );
                        }

                        return Ok((eigenvalues, vr_complex, vl_complex));
                    }
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

macro_rules! implement_evd_complex {
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

                let mut w = vec![<$scalar as Zero>::zero(); m];

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
                    EigenvectorMode::All => {
                        jobvr = b'V';
                        vr_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vr_data = vr_matrix.as_mut().unwrap().data_mut();
                        jobvl = b'V';
                        vl_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vl_data = vl_matrix.as_mut().unwrap().data_mut();
                    }
                    EigenvectorMode::Right => {
                        jobvr = b'V';
                        vr_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vr_data = vr_matrix.as_mut().unwrap().data_mut();
                        jobvl = b'N';
                        vl_matrix = None;
                        vl_data = vl_dummy.as_mut_slice();
                    }
                    EigenvectorMode::Left => {
                        jobvl = b'V';
                        vl_matrix = Some(rlst_dense::rlst_mat![$scalar, (m, m)]);
                        vl_data = vl_matrix.as_mut().unwrap().data_mut();
                        jobvr = b'N';
                        vr_matrix = None;
                        vr_data = vr_dummy.as_mut_slice();
                    }
                    EigenvectorMode::None => {
                        jobvr = b'N';
                        vr_matrix = None;
                        vr_data = vr_dummy.as_mut_slice();
                        jobvl = b'N';
                        vl_matrix = None;
                        vl_data = vl_dummy.as_mut_slice();
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
                        w.as_mut_slice(),
                        vl_data,
                        n as i32,
                        vr_data,
                        n as i32,
                    )
                };

                match info {
                    0 => Ok((w, vr_matrix, vl_matrix)),
                    _ => Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

implement_evd_real!(f64, dgeev);
implement_evd_real!(f32, sgeev);
implement_evd_complex!(c32, cgeev);
implement_evd_complex!(c64, zgeev);

#[cfg(test)]
mod test {

    use super::*;
    use crate::linalg::LinAlg;
    use crate::traits::norm2::Norm2;
    use approx::assert_relative_eq;
    use paste::paste;
    use rlst_dense::Dot;

    macro_rules! impl_test_ev {
        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                pub fn [<test_ev_$scalar>]() {
                    let mut rlst_mat = rlst_dense::rlst_mat![$scalar, (2, 2)];
                    rlst_mat.fill_from_seed_equally_distributed(0);

                    let (eigvals, right, left) =
                        rlst_mat.linalg().evd(EigenvectorMode::All).unwrap();
                    let left = left.unwrap().conj().transpose().eval();
                    let right = right.unwrap();

                    let mut diag = rlst_dense::rlst_mat![<$scalar as Scalar>::Complex, (2, 2)];
                    diag.set_diag_from_slice(eigvals.as_slice());

                    let rlst_mat = rlst_mat.to_complex().eval();

                    let res = (rlst_mat.dot(&right) - right.dot(&diag))
                        .eval()
                        .linalg()
                        .norm2()
                        .unwrap();

                    assert!(res < $tol);

                    let res = (left.dot(&rlst_mat) - diag.dot(&left))
                        .eval()
                        .linalg()
                        .norm2()
                        .unwrap();

                    assert!(res < $tol);
                }
            }
        };
    }

    #[test]
    fn test_complex_conjugate_pair() {
        let mut mat = rlst_dense::rlst_mat![f64, (3, 3)];

        mat[[0, 0]] = 1.0;
        mat[[0, 1]] = 1.0;
        mat[[1, 0]] = -1.0;
        mat[[1, 1]] = 1.0;

        mat[[2, 2]] = 4.0;

        let eigvals = mat.linalg().eigenvalues().unwrap();

        assert_relative_eq!(eigvals[0], c64::new(1.0, 1.0), epsilon = 1E-12);
        assert_relative_eq!(eigvals[1], c64::new(1.0, -1.0), epsilon = 1E-12);
        assert_relative_eq!(eigvals[2], c64::new(4.0, 0.0), epsilon = 1E-12);
    }

    impl_test_ev!(f64, 1E-12);
    impl_test_ev!(f32, 1E-5);
    impl_test_ev!(c64, 1E-12);
    impl_test_ev!(c32, 1E-5);
}
