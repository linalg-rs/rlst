//! Singular Value Decomposition.
use crate::array::Array;
use crate::traits::*;
use crate::types::{c32, c64, RlstError, RlstResult, RlstScalar};
use lapack::{cgesvd, dgesvd, sgesvd, zgesvd};
use num::traits::Zero;

use super::assert_lapack_stride;

/// Singular value decomposition
pub trait MatrixSvd {
    /// Item type
    type Item: RlstScalar;

    /// Compute the singular values of the matrix.
    ///
    /// For a `(m, n)` matrix A the slice `singular_values` has
    /// length `k=min(m, n)`.
    ///
    /// This method allocates temporary memory during execution.
    fn into_singular_values_alloc(
        self,
        singular_values: &mut [<Self::Item as RlstScalar>::Real],
    ) -> RlstResult<()>;

    /// Compute the singular value decomposition.
    ///
    /// We assume that `A` is a `(m, n)` matrix and assume
    /// `k=min(m, n)`. This method computes the singular value
    /// decomposition `A = USVt`.
    ///
    /// # Parameters
    ///
    /// - `u` - Stores the matrix `U`. For the full SVD the shape
    ///         needs to be `(m, m)`. For the reduced SVD it needs to be `(m, k)`.
    /// - `vt` - Stores the matrix `Vt`. For the full SVD the shape needs to be `(n, n)`.
    ///          For the reduced SVD it needs to be `(k, n)`. Note that `vt` stores
    ///          the complex conjugate transpose of the matrix of right singular vectors.
    ///          Hence, the columns of `vt.transpose().conj()` will be the right singular vectors.
    /// - `singular_values` - Stores the `k` singular values of `A`.
    /// - `mode` - Choose between full SVD [SvdMode::Full] or reduced SVD [SvdMode::Reduced].
    ///
    /// This method allocates temporary memory during execution.
    fn into_svd_alloc<
        ArrayImplU: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
        ArrayImplVt: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        self,
        u: Array<Self::Item, ArrayImplU, 2>,
        vt: Array<Self::Item, ArrayImplVt, 2>,
        singular_values: &mut [<Self::Item as RlstScalar>::Real],
        mode: SvdMode,
    ) -> RlstResult<()>;
}

/// SVD mode
pub enum SvdMode {
    /// Reduces SVD
    Reduced,
    /// Full SVD
    Full,
}

macro_rules! impl_svd_real {
    ($scalar:ty, $gesvd:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixSvd for Array<$scalar, ArrayImpl, 2>
        {
            type Item = $scalar;

            fn into_singular_values_alloc(
                mut self,
                singular_values: &mut [<$scalar as RlstScalar>::Real],
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                assert!(!self.is_empty(), "Matrix is empty.");

                assert_lapack_stride(self.stride());
                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);
                let lda = self.stride()[1] as i32;
                let mut u = [<$scalar as Zero>::zero(); 1];
                let mut vt = [<$scalar as Zero>::zero(); 1];
                let ldu = 1;
                let ldvt = 1;
                let mut info = 0;

                unsafe {
                    $gesvd(
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

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
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

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }

            fn into_svd_alloc<
                ArrayImplU: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplVt: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut self,
                mut u: Array<$scalar, ArrayImplU, 2>,
                mut vt: Array<$scalar, ArrayImplVt, 2>,
                singular_values: &mut [<$scalar as RlstScalar>::Real],
                mode: SvdMode,
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                let jobu;
                let jobvt;

                assert!(!self.is_empty(), "Matrix is empty.");

                assert_lapack_stride(self.stride());
                assert_lapack_stride(u.stride());
                assert_lapack_stride(vt.stride());

                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);

                match mode {
                    SvdMode::Full => {
                        jobu = b'A';
                        jobvt = b'A';

                        assert_eq!(u.shape(), [m as usize, m as usize]);
                        assert_eq!(vt.shape(), [n as usize, n as usize]);
                    }
                    SvdMode::Reduced => {
                        jobu = b'S';
                        jobvt = b'S';

                        assert_eq!(u.shape(), [m as usize, k as usize]);
                        assert_eq!(vt.shape(), [k as usize, n as usize]);
                    }
                }

                let lda = self.stride()[1] as i32;
                let ldu = u.stride()[1] as i32;
                let ldvt = vt.stride()[1] as i32;
                let mut info = 0;

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

macro_rules! impl_svd_complex {
    ($scalar:ty, $gesvd:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixSvd for Array<$scalar, ArrayImpl, 2>
        {
            type Item = $scalar;

            fn into_singular_values_alloc(
                mut self,
                singular_values: &mut [<$scalar as RlstScalar>::Real],
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                assert!(!self.is_empty(), "Matrix is empty.");

                assert_lapack_stride(self.stride());
                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);
                let lda = self.stride()[1] as i32;
                let mut u = [<$scalar as Zero>::zero(); 1];
                let mut vt = [<$scalar as Zero>::zero(); 1];
                let ldu = 1;
                let ldvt = 1;
                let mut info = 0;

                let mut rwork =
                    vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 5 * k as usize];

                unsafe {
                    $gesvd(
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
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
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
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }

            fn into_svd_alloc<
                ArrayImplU: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplVt: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut self,
                mut u: Array<$scalar, ArrayImplU, 2>,
                mut vt: Array<$scalar, ArrayImplVt, 2>,
                singular_values: &mut [<$scalar as RlstScalar>::Real],
                mode: SvdMode,
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                let jobu;
                let jobvt;

                assert!(!self.is_empty(), "Matrix is empty.");

                assert_lapack_stride(self.stride());
                assert_lapack_stride(u.stride());
                assert_lapack_stride(vt.stride());

                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);

                match mode {
                    SvdMode::Full => {
                        jobu = b'A';
                        jobvt = b'A';

                        assert_eq!(u.shape(), [m as usize, m as usize]);
                        assert_eq!(vt.shape(), [n as usize, n as usize]);
                    }
                    SvdMode::Reduced => {
                        jobu = b'S';
                        jobvt = b'S';

                        assert_eq!(u.shape(), [m as usize, k as usize]);
                        assert_eq!(vt.shape(), [k as usize, n as usize]);
                    }
                }

                let lda = self.stride()[1] as i32;
                let ldu = u.stride()[1] as i32;
                let ldvt = vt.stride()[1] as i32;
                let mut info = 0;

                let mut rwork =
                    vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 5 * k as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

impl_svd_real!(f64, dgesvd);
impl_svd_real!(f32, sgesvd);
impl_svd_complex!(c32, cgesvd);
impl_svd_complex!(c64, zgesvd);
