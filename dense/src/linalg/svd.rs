//! Implement the SVD
use crate::array::Array;
use lapack::dgesvd;
use num::traits::Zero;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};

use super::assert_lapack_stride;

pub enum SvdMode<
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = f64> + Stride<2> + Shape<2> + RawAccessMut<Item = f64>,
> {
    All(Array<f64, ArrayImpl, 2>),
    Slim(Array<f64, ArrayImpl, 2>),
    None,
}

impl<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = f64> + Stride<2> + Shape<2> + RawAccessMut<Item = f64>,
    > Array<f64, ArrayImpl, 2>
{
    pub fn into_singular_values(mut self, singular_values: &mut [f64]) -> RlstResult<()> {
        let lwork: i32 = -1;
        let mut work = [<f64 as Zero>::zero(); 1];

        assert_lapack_stride(self.stride());
        let m = self.shape()[0] as i32;
        let n = self.shape()[1] as i32;
        let k = std::cmp::min(m, n);
        assert_eq!(k, singular_values.len() as i32);
        let lda = self.stride()[1] as i32;
        let mut u = [<f64 as Zero>::zero(); 1];
        let mut vt = [<f64 as Zero>::zero(); 1];
        let ldu = 1;
        let ldvt = 1;
        let mut info = 0;

        unsafe {
            dgesvd(
                b'N',
                b'N',
                m,
                n,
                self.data_mut(),
                lda,
                singular_values,
                &mut u,
                ldu,
                &mut vt,
                ldvt,
                &mut work,
                lwork,
                &mut info,
            );
        }

        let lwork = work[0] as i32;
        let mut work = vec![<f64 as Zero>::zero(); lwork as usize];

        unsafe {
            dgesvd(
                b'N',
                b'N',
                m,
                n,
                self.data_mut(),
                lda,
                singular_values,
                &mut u,
                ldu,
                &mut vt,
                ldvt,
                &mut work,
                lwork,
                &mut info,
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::*;

    use rlst_common::tools::PrettyPrint;

    use crate::array::empty_array;
    use crate::rlst_dynamic_array2;

    #[test]
    fn test_singular_values() {
        let shape = [20, 9];
        let mut mat = rlst_dynamic_array2!(f64, [shape[0], shape[0]]);
        let mut q = rlst_dynamic_array2!(f64, [shape[0], shape[0]]);
        let mut sigma = rlst_dynamic_array2!(f64, shape);

        mat.fill_from_seed_equally_distributed(0);
        let qr = mat.into_qr().unwrap();
        qr.get_q(q.view_mut()).unwrap();

        for index in 0..shape[1] {
            sigma[[index, index]] = (shape[1] - index) as f64;
        }

        let a = empty_array::<f64, 2>().simple_mult_into_resize(q.view(), sigma.view());

        let mut singvals = rlst_dynamic_array2!(f64, [shape[1], 1]);

        let _ = a.into_singular_values(singvals.data_mut());
        singvals.pretty_print();
    }
}

// macro_rules! implement_svd {
//     ($scalar:ty, $lapack_gesvd:ident) => {
//         impl Svd for DenseMatrixLinAlgBuilder<$scalar> {
//             type T = $scalar;
//             fn svd(
//                 self,
//                 u_mode: Mode,
//                 vt_mode: Mode,
//             ) -> RlstResult<(
//                 Vec<<$scalar as Scalar>::Real>,
//                 Option<MatrixD<$scalar>>,
//                 Option<MatrixD<$scalar>>,
//             )> {
//                 let mut mat = self.mat;

//                 let m = mat.shape().0 as i32;
//                 let n = mat.shape().1 as i32;
//                 let k = std::cmp::min(m, n);
//                 let lda = mat.stride().1 as i32;

//                 let mut s_values = vec![<<$scalar as Scalar>::Real as Zero>::zero(); k as usize];
//                 let mut superb =
//                     vec![<<$scalar as Scalar>::Real as Zero>::zero(); (k - 1) as usize];
//                 let jobu;
//                 let jobvt;
//                 let mut u_matrix;
//                 let mut vt_matrix;
//                 let ldu;
//                 let ldvt;
//                 let u_data: &mut [Self::T];
//                 let vt_data: &mut [Self::T];

//                 // The following two are needed as dummy arrays for the case
//                 // that the computation of u and v is not requested. Even then
//                 // we still need to pass a valid reference to the Lapack routine.
//                 let mut u_dummy = vec![<Self::T as Zero>::zero(); 1];
//                 let mut vt_dummy = vec![<Self::T as Zero>::zero(); 1];

//                 match u_mode {
//                     Mode::All => {
//                         jobu = b'A';
//                         u_matrix = Some(rlst_dynamic_mat![$scalar, (m as usize, m as usize)]);
//                         u_data = u_matrix.as_mut().unwrap().data_mut();
//                         ldu = m as i32;
//                     }
//                     Mode::Slim => {
//                         jobu = b'S';
//                         u_matrix = Some(rlst_dynamic_mat![$scalar, (m as usize, k as usize)]);
//                         u_data = u_matrix.as_mut().unwrap().data_mut();
//                         ldu = m as i32;
//                     }
//                     Mode::None => {
//                         jobu = b'N';
//                         u_matrix = None;
//                         u_data = u_dummy.as_mut_slice();
//                         ldu = m as i32;
//                     }
//                 };

//                 match vt_mode {
//                     Mode::All => {
//                         jobvt = b'A';
//                         vt_matrix = Some(rlst_dynamic_mat![$scalar, (n as usize, n as usize)]);
//                         vt_data = vt_matrix.as_mut().unwrap().data_mut();
//                         ldvt = n as i32;
//                     }
//                     Mode::Slim => {
//                         jobvt = b'S';
//                         vt_matrix = Some(rlst_dynamic_mat![$scalar, (k as usize, n as usize)]);
//                         vt_data = vt_matrix.as_mut().unwrap().data_mut();
//                         ldvt = k as i32;
//                     }
//                     Mode::None => {
//                         jobvt = b'N';
//                         vt_matrix = None;
//                         vt_data = vt_dummy.as_mut_slice();
//                         ldvt = k as i32;
//                     }
//                 }

//                 let info = unsafe {
//                     lapacke::$lapack_gesvd(
//                         lapacke::Layout::ColumnMajor,
//                         jobu,
//                         jobvt,
//                         m,
//                         n,
//                         mat.data_mut(),
//                         lda,
//                         s_values.as_mut_slice(),
//                         u_data,
//                         ldu,
//                         vt_data,
//                         ldvt,
//                         superb.as_mut_slice(),
//                     )
//                 };

//                 match info {
//                     0 => return Ok((s_values, u_matrix, vt_matrix)),
//                     _ => return Err(RlstError::LapackError(info)),
//                 }
//             }
//         }
//     };
// }

// implement_svd!(f64, dgesvd);
// implement_svd!(f32, sgesvd);
// implement_svd!(c32, cgesvd);
// implement_svd!(c64, zgesvd);

// #[cfg(test)]
// mod test {
//     use approx::assert_relative_eq;
//     use rand::SeedableRng;

//     use super::*;
//     use crate::linalg::LinAlg;
//     use rand_chacha::ChaCha8Rng;
//     use rlst_dense::Dot;

//     #[test]
//     fn test_thick_svd() {
//         let mut rng = ChaCha8Rng::seed_from_u64(0);

//         let m = 3;
//         let n = 4;
//         let k = std::cmp::min(m, n);
//         let mut mat = rlst_dynamic_mat!(f64, (m, n));

//         mat.fill_from_equally_distributed(&mut rng);
//         let expected = mat.copy();

//         let (singular_values, u_matrix, vt_matrix) =
//             mat.linalg().svd(Mode::Slim, Mode::Slim).unwrap();

//         let u_matrix = u_matrix.unwrap();
//         let vt_matrix = vt_matrix.unwrap();

//         let mut sigma = rlst_dynamic_mat!(f64, (k, k));
//         for index in 0..k {
//             sigma[[index, index]] = singular_values[index];
//         }

//         let tmp = sigma.dot(&vt_matrix);

//         let actual = u_matrix.dot(&tmp);

//         for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
//             assert_relative_eq!(a, e, epsilon = 1E-13);
//         }
//     }

//     #[test]
//     fn test_thin_svd() {
//         let mut rng = ChaCha8Rng::seed_from_u64(0);

//         let m = 4;
//         let n = 3;
//         let k = std::cmp::min(m, n);
//         let mut mat = rlst_dynamic_mat!(f64, (m, n));

//         mat.fill_from_equally_distributed(&mut rng);
//         let expected = mat.copy();

//         let (singular_values, u_matrix, vt_matrix) =
//             mat.linalg().svd(Mode::Slim, Mode::Slim).unwrap();

//         let u_matrix = u_matrix.unwrap();
//         let vt_matrix = vt_matrix.unwrap();

//         let mut sigma = rlst_dynamic_mat!(f64, (k, k));
//         for index in 0..k {
//             sigma[[index, index]] = singular_values[index];
//         }

//         let tmp = sigma.dot(&vt_matrix);

//         let actual = u_matrix.dot(&tmp);

//         for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
//             assert_relative_eq!(a, e, epsilon = 1E-13);
//         }
//     }

//     #[test]
//     fn test_full_svd() {
//         let mut rng = ChaCha8Rng::seed_from_u64(0);

//         let m = 5;
//         let n = 5;
//         let k = std::cmp::min(m, n);
//         let mut mat = rlst_dynamic_mat!(f64, (m, n));

//         mat.fill_from_equally_distributed(&mut rng);
//         let expected = mat.copy();

//         let (singular_values, u_matrix, vt_matrix) =
//             mat.linalg().svd(Mode::All, Mode::All).unwrap();

//         let u_matrix = u_matrix.unwrap();
//         let vt_matrix = vt_matrix.unwrap();

//         let mut sigma = rlst_dynamic_mat!(f64, (k, k));
//         for index in 0..k {
//             sigma[[index, index]] = singular_values[index];
//         }

//         let tmp = sigma.dot(&vt_matrix);

//         let actual = u_matrix.dot(&tmp);

//         for (a, e) in actual.iter_col_major().zip(expected.iter_col_major()) {
//             assert_relative_eq!(a, e, epsilon = 1E-13);
//         }
//     }
// }
