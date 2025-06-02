//! Matrix inverse.
//!
//!
use super::LapackWrapper;
use crate::dense::traits::{RawAccessMut, Shape, Stride};
use crate::{
    dense::types::{c32, c64, RlstError, RlstResult, RlstScalar},
    Array,
};
use lapack::{cgetrf, cgetri, dgetrf, dgetri, sgetrf, sgetri, zgetrf, zgetri};
use num::traits::Zero;

/// A trait for computing the inverse of a matrix in place.
pub trait LapackInverse {
    /// Compute the matrix inverse in place.
    fn inverse(&mut self) -> RlstResult<()>;
}

macro_rules! impl_inverse {
    ($scalar:ty, $getrf: expr, $getri:expr) => {
        impl<ArrayImpl: RawAccessMut<Item = $scalar> + Shape<2> + Stride<2>> LapackInverse
            for LapackWrapper<$scalar, ArrayImpl>
        {
            /// Compute the matrix inverse in place.
            fn inverse(&mut self) -> RlstResult<()> {
                let (m, n, lda) = (self.m, self.n, self.lda);

                assert!(self.m * self.n != 0, "Matrix is empty.");

                assert_eq!(m, n);

                let mut ipiv = vec![0; m as usize];

                let mut lwork = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                let mut info = 0;

                unsafe { $getrf(m, m, self.arr.data_mut(), lda, &mut ipiv, &mut info) }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                unsafe {
                    $getri(
                        n,
                        self.arr.data_mut(),
                        lda,
                        &ipiv,
                        &mut work,
                        lwork,
                        &mut info,
                    )
                };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                lwork = work[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $getri(
                        n,
                        self.arr.data_mut(),
                        lda,
                        &ipiv,
                        &mut work,
                        lwork,
                        &mut info,
                    )
                };

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
