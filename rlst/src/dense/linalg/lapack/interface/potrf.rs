//! Implementation of ?potrf - Cholesky factorization of a symmetric positive-definite matrix.

use lapack::{cpotrf, dpotrf, spotrf, zpotrf};

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64};

use crate::dense::linalg::lapack::interface::lapack_return;

/// `Uplo` parameter for `?potrf` to specify which triangular part of the matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PotrfUplo {
    /// Upper triangular part of the matrix is stored.
    Upper = b'U',
    /// Lower triangular part of the matrix is stored.
    Lower = b'L',
}

/// ?potrf - Cholesky factorization of a symmetric positive-definite matrix.
pub trait Potrf: Sized {
    /// Perform Cholesky factorization of a symmetric positive-definite matrix.
    ///
    /// If `uplo` is `Upper`, the factorization is of the form:
    /// A = U^H * U, where U is an upper triangular matrix. If `uplo` is `Lower`, the
    /// factorization is of the form:
    /// A = L * L^H, where L is a lower triangular matrix.
    ///
    /// **Arguments:**
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    /// - `n`: The order of the matrix A.
    /// - `a`: The matrix A to be factored.
    /// - `lda`: The leading dimension of the matrix A.
    ///
    /// **Returns:**
    /// A `LapackResult<()>` indicating success or failure.
    fn potrf(uplo: PotrfUplo, n: usize, a: &mut [Self], lda: usize) -> LapackResult<()>;
}

macro_rules! implement_potrf {
    ($scalar:ty, $potrf:expr) => {
        impl Potrf for $scalar {
            fn potrf(uplo: PotrfUplo, n: usize, a: &mut [Self], lda: usize) -> LapackResult<()> {
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, n),
                    "Require `lda` {} >= `max(1, n)` {}.",
                    lda,
                    std::cmp::max(1, n)
                );

                let mut info = 0;

                unsafe {
                    $potrf(uplo as u8, n as i32, a, lda as i32, &mut info);
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_potrf!(f32, spotrf);
implement_potrf!(f64, dpotrf);
implement_potrf!(c32, cpotrf);
implement_potrf!(c64, zpotrf);
