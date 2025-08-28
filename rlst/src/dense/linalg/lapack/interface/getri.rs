//! Implementation of ?getri - Matrix inversion using LU factorization.

use lapack::{cgetri, dgetri, sgetri, zgetri};

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64, LapackError};

use crate::dense::linalg::lapack::interface::lapack_return;

use num::{complex::ComplexFloat, Zero};

/// ?getri - Matrix inversion using LU factorization.
///
/// **Arguments:**
///
/// - `n`: Number of rows and columns in the square matrix `a`.
/// - `a`: The factors `L` and `U` from the LU factorization of the matrix.
/// - `ipiv`: Pivot indices from the LU factorization.
/// - `lda`: Leading dimension of the matrix `a`.
///
/// **Returns:**
/// - The Lapack info code if the inversion fails.
pub trait Getri: Sized {
    /// Compute the inverse of a matrix `a` using the LU factors and pivot indices.
    fn getri(n: usize, a: &mut [Self], lda: usize, ipiv: &[i32]) -> LapackResult<()>;
}

macro_rules! implement_getri {
    ($scalar:ty, $getri:expr) => {
        impl Getri for $scalar {
            fn getri(n: usize, a: &mut [Self], lda: usize, ipiv: &[i32]) -> LapackResult<()> {
                let mut info = 0;

                assert!(
                    lda >= std::cmp::max(1, n),
                    "Require `lda` {} >= `max(1, N)` {} .",
                    lda,
                    std::cmp::max(1, n)
                );
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n,
                );

                assert_eq!(
                    ipiv.len(),
                    n,
                    "Require `ipiv.len()` {} == `n` {}.",
                    ipiv.len(),
                    n,
                );

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $getri(n as i32, a, lda as i32, ipiv, &mut work, -1, &mut info);
                }

                let lwork = work[0].re() as i32;

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $getri(n as i32, a, lda as i32, ipiv, &mut work, lwork, &mut info);
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_getri!(f32, sgetri);
implement_getri!(f64, dgetri);
implement_getri!(c32, cgetri);
implement_getri!(c64, zgetri);
