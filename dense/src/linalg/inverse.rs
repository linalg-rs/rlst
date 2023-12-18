//! Matrix inverse.
//!
//!
use crate::array::Array;
use crate::traits::*;
use lapack::{cgetrf, cgetri, dgetrf, dgetri, sgetrf, sgetri, zgetrf, zgetri};
use num::traits::Zero;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};

use super::assert_lapack_stride;

macro_rules! impl_inverse {
    ($scalar:ty, $getrf: expr, $getri:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > Array<$scalar, ArrayImpl, 2>
        {
            //! Compute the matrix inverse.
            //!
            //! The matrix inverse is defined for a two dimensional square array `arr` of
            //! shape `[m, m]`.
            //!
            //! # Example
            //!
            //! The following command computes the inverse of an array `a`. The content
            //! of `a` is replaced by the inverse.
            //! ```
            //! # use rlst_dense::rlst_dynamic_array2;
            //! # let mut a = rlst_dynamic_array2!(f64, [3, 3]);
            //! # a.fill_from_seed_equally_distributed(0);
            //! a.view_mut().into_inverse_alloc().unwrap();
            //! ```
            //! This method allocates memory for the inverse computation.
            pub fn into_inverse_alloc(mut self) -> RlstResult<()> {
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

#[cfg(test)]
mod test {

    use super::*;

    use crate::assert_array_abs_diff_eq;
    use paste::paste;

    use crate::array::empty_array;
    use crate::rlst_dynamic_array2;

    macro_rules! impl_inverse_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

                #[test]
                fn [<test_inverse_$scalar>]() {
                    let n = 4;

                    let mut a = rlst_dynamic_array2!($scalar, [n, n]);
                    let mut b = rlst_dynamic_array2!($scalar, [n, n]);

                    let mut ident = rlst_dynamic_array2!($scalar, [n, n]);
                    ident.set_identity();

                    a.fill_from_seed_equally_distributed(0);
                    b.fill_from(a.view());

                    a.view_mut().into_inverse_alloc().unwrap();

                    let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(a.view(), b.view());

                    assert_array_abs_diff_eq!(actual, ident, $tol);
                }

            }
        };
    }

    impl_inverse_tests!(f64, 1E-12);
    impl_inverse_tests!(f32, 5E-6);
    impl_inverse_tests!(c32, 5E-6);
    impl_inverse_tests!(c64, 1E-12);
}
