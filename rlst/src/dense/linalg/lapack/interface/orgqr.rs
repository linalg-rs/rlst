//! Implementation of ?orgqr - Compute the matrix Q from the QR factorization of a matrix A.

use lapack::{cungqr, dorgqr, sorgqr, zungqr};

use crate::dense::linalg::lapack::interface::lapack_return;

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64, LapackError};

use num::{complex::ComplexFloat, Zero};

/// Compute the matrix Q from the QR factorization of a matrix A.
///
/// **Arguments:**
/// - `m`: The number of rows in the matrix Q.
/// - `n`: The number of columns in the matrix Q.
/// - `k`: The number of elementary reflectors.
/// - `a`: The matrix containing the elementary reflectors.
/// - `lda`: The leading dimension of the matrix A.
/// - `tau`: The scalar factors of the elementary reflectors.
///
/// **Returns:**
/// A `LapackResult<()>` indicating success or failure.
pub trait Orgqr {
    /// Compute the matrix Q from the QR factorization of a matrix A.
    fn orgqr(
        m: usize,
        n: usize,
        k: usize,
        a: &mut [Self],
        lda: usize,
        tau: &[Self],
    ) -> LapackResult<()>
    where
        Self: Sized;
}

macro_rules! implement_orgqr {
    ($scalar:ty, $orgqr:expr) => {
        impl Orgqr for $scalar {
            fn orgqr(
                m: usize,
                n: usize,
                k: usize,
                a: &mut [Self],
                lda: usize,
                tau: &[Self],
            ) -> LapackResult<()> {
                let mut info = 0;

                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    lda * n,
                    a.len()
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );

                assert_eq!(
                    tau.len(),
                    k,
                    "Require `tau.len()` {} == `k` {}.",
                    tau.len(),
                    k
                );

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $orgqr(
                        m as i32, n as i32, k as i32, a, lda as i32, tau, &mut work, -1, &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;

                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $orgqr(
                        m as i32,
                        n as i32,
                        k as i32,
                        a,
                        lda as i32,
                        tau,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_orgqr!(f32, sorgqr);
implement_orgqr!(f64, dorgqr);
implement_orgqr!(c32, cungqr);
implement_orgqr!(c64, zungqr);
