//! Matrix inverse.
//!
//!
use super::LapackArrayMut;
use crate::dense::types::{c32, c64, RlstError, RlstResult, RlstScalar};
use lapack::{cgetrf, cgetri, dgetrf, dgetri, sgetrf, sgetri, zgetrf, zgetri};
use num::traits::Zero;

macro_rules! impl_inverse {
    ($scalar:ty, $getrf: expr, $getri:expr) => {
        impl LapackArrayMut<'_, $scalar> {
            /// Compute the matrix inverse in place.
            pub fn inverse(&mut self) -> RlstResult<()> {
                let m = self.m;
                let n = self.n;

                assert!(self.m * self.n != 0, "Matrix is empty.");

                assert_eq!(m, n);

                let lda = self.lda;
                let mut ipiv = vec![0; m as usize];

                let mut lwork = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                let mut info = 0;

                unsafe { $getrf(m, m, self.data, lda, &mut ipiv, &mut info) }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                unsafe { $getri(n, self.data, lda, &ipiv, &mut work, lwork, &mut info) };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                lwork = work[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe { $getri(n, self.data, lda, &ipiv, &mut work, lwork, &mut info) };

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

impl_inverse!(f64, dgetrf, dgetri);
impl_inverse!(f32, sgetrf, sgetri);
impl_inverse!(c32, cgetrf, cgetri);
impl_inverse!(c64, zgetrf, zgetri);
