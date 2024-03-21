//! Matrix inverse.
//!
//!
use crate::array::Array;
use crate::traits::{RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue};
use crate::types::{c32, c64, RlstError, RlstResult, RlstScalar};
use lapack::{cgetrf, cgetri, dgetrf, dgetri, sgetrf, sgetri, zgetrf, zgetri};
use num::traits::Zero;

use super::assert_lapack_stride;

/// Compute the matrix inverse.
///
/// The matrix inverse is defined for a two dimensional square array `arr` of
/// shape `[m, m]`.
///
/// # Example
///
/// The following command computes the inverse of an array `a`. The content
/// of `a` is replaced by the inverse.
/// ```ignore
/// # use rlst_dense::rlst_dynamic_array2;
/// # use rlst_dense::linalg::inverse::MatrixInverse;
/// # let mut a = rlst_dynamic_array2!(f64, [3, 3]);
/// # a.fill_from_seed_equally_distributed(0);
/// a.view_mut().into_inverse_alloc().unwrap();
/// ```
/// This method allocates memory for the inverse computation.
pub trait MatrixInverse {
    /// Compute the matrix inverse
    fn into_inverse_alloc(self) -> RlstResult<()>;
}

macro_rules! impl_inverse {
    ($scalar:ty, $getrf: expr, $getri:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixInverse for Array<$scalar, ArrayImpl, 2>
        {
            fn into_inverse_alloc(mut self) -> RlstResult<()> {
                assert_lapack_stride(self.stride());

                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;

                assert!(!self.is_empty(), "Matrix is empty.");

                assert_eq!(m, n);

                let lda = self.stride()[1] as i32;
                let mut ipiv = vec![0; m as usize];

                let mut lwork = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                let mut info = 0;

                unsafe { $getrf(m, m, self.data_mut(), lda, &mut ipiv, &mut info) }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                unsafe { $getri(n, self.data_mut(), lda, &ipiv, &mut work, lwork, &mut info) };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                lwork = work[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe { $getri(n, self.data_mut(), lda, &ipiv, &mut work, lwork, &mut info) };

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
