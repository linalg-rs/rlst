//! Implementation of ?getrf - LU factorization with partial pivoting

use lapack::{cgetrf, dgetrf, sgetrf, zgetrf};

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64, LapackError};

/// ?getrf - LU factorization with partial pivoting.
///
/// **Arguments:**
///
/// - `m`: Number of rows in the matrix `a`.
/// - `n`: Number of columns in the matrix `a`.
/// - `a`: The matrix to be factored, stored in column-major order.
/// - `ipiv`: Output vector of pivot indices, which indicates the row swaps performed during.
///   `ipiv` must have length `min(m, n)`.
/// - `lda`: Leading dimension of the matrix `a`.
///
pub trait Getrf: Sized {
    /// Perform LU factorization of a matrix `a` with dimensions `m` x `n`.
    fn getrf(m: usize, n: usize, a: &mut [Self], lda: usize, ipiv: &mut [i32]) -> LapackResult<()>;
}

macro_rules! impl_getrf {
    ($scalar:ty, $getrf:expr) => {
        impl Getrf for $scalar {
            fn getrf(
                m: usize,
                n: usize,
                a: &mut [$scalar],
                lda: usize,
                ipiv: &mut [i32],
            ) -> LapackResult<()> {
                let k = std::cmp::min(m, n);
                let mut info = 0;

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, M)` {} .",
                    lda,
                    std::cmp::max(1, m)
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
                    k,
                    "Require `ipiv.len()` {} == `min(m, n)` {}.",
                    ipiv.len(),
                    k,
                );

                unsafe {
                    $getrf(m as i32, n as i32, a, lda as i32, ipiv, &mut info);
                }

                if info != 0 {
                    Err(LapackError::LapackInfoCode(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

impl_getrf!(f32, sgetrf);
impl_getrf!(f64, dgetrf);
impl_getrf!(c32, cgetrf);
impl_getrf!(c64, zgetrf);
