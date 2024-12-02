//! Matrix inverse.
//!
//!
use crate::dense::array::Array;
use crate::dense::traits::{RawAccessMut, Shape, Stride, UnsafeRandomAccessByValue};
use crate::dense::types::{c32, c64, RlstError, RlstResult, RlstScalar};
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
/// ```
/// # use rlst::rlst_dynamic_array2;
/// # use rlst::dense::linalg::inverse::MatrixInverse;
/// # let mut a = rlst_dynamic_array2!(f64, [3, 3]);
/// # a.fill_from_seed_equally_distributed(0);
/// a.view_mut().into_inverse_alloc().unwrap();
/// ```
/// This method allocates memory for the inverse computation.
pub trait MatrixInverse: RlstScalar {
    /// Compute the matrix inverse
    fn into_inverse_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
    ) -> RlstResult<()>;
}

macro_rules! impl_inverse {
    ($scalar:ty, $getrf: expr, $getri:expr) => {
        impl MatrixInverse for $scalar {
            fn into_inverse_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut arr: Array<Self, ArrayImpl, 2>,
            ) -> RlstResult<()> {
                assert_lapack_stride(arr.stride());

                let m = arr.shape()[0] as i32;
                let n = arr.shape()[1] as i32;

                assert!(!arr.is_empty(), "Matrix is empty.");

                assert_eq!(m, n);

                let lda = arr.stride()[1] as i32;
                let mut ipiv = vec![0; m as usize];

                let mut lwork = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                let mut info = 0;

                unsafe { $getrf(m, m, arr.data_mut(), lda, &mut ipiv, &mut info) }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                unsafe { $getri(n, arr.data_mut(), lda, &ipiv, &mut work, lwork, &mut info) };

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                lwork = work[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe { $getri(n, arr.data_mut(), lda, &ipiv, &mut work, lwork, &mut info) };

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

impl<
        Item: RlstScalar + MatrixInverse,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
    /// Return the matrix inverse.
    pub fn into_inverse_alloc(self) -> RlstResult<()> {
        <Item as MatrixInverse>::into_inverse_alloc(self)
    }
}
