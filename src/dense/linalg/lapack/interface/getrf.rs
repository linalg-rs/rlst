//! Implementation of ?getrf - LU factorization with partial pivoting

use lapack::{cgetrf, dgetrf, sgetrf, zgetrf};

use super::{c32, c64, LapackError, LapackResult};

/// ?getrf - LU factorization with partial pivoting.
///
/// **Arguments:**
///
/// - `m`: Number of rows in the matrix `a`.
/// - `n`: Number of columns in the matrix `a`.
/// - `a`: The matrix to be factored, stored in column-major order.
/// - `lda`: Leading dimension of the matrix `a`.
///
/// **Returns:**
/// - A result containing a vector `ipiv` of pivot indices if successful, or an error if the factorization
/// fails.
pub trait Getrf: Sized {
    /// Perform LU factorization of a matrix `a` with dimensions `m` x `n`.
    fn getrf(m: usize, n: usize, a: &mut [Self], lda: usize) -> LapackResult<Vec<i32>>;
}

macro_rules! impl_getrf {
    ($scalar:ty, $getrf:expr) => {
        impl Getrf for $scalar {

            fn getrf(
                m: usize,
                n: usize,
                a: &mut [$scalar],
                lda: usize,
            ) -> LapackResult<Vec<i32>> {
                let k = std::cmp::min(m, n);
                let mut ipiv = vec![0 as i32; k];
                let mut info = 0;

                assert!(lda >= std::cmp::max(1, m), "Require `lda` {} >= `max(1, M)` {} .", lda, std::cmp::max(1, m));
                assert_eq!(
                    a.len(),
                    lda * m,
                    "Require `a.len()` {} == `lda * m` {}.",
                    a.len(),
                    lda * m,
                );

                assert_eq!(
                    a.len(),
                    lda * n,
                    "The length of the matrix `a` must be equal to `lda * n`.: a.len {} != {} lda * n",
                    a.len(),
                    lda * n,
                );


                unsafe {
                    $getrf(m as i32, n as i32, a, lda as i32, &mut ipiv, &mut info);
                }

                if info != 0 {
                    Err(LapackError::LapackInfoCode(info))
                } else {
                    Ok(ipiv)
                }
            }
        }
    };
}

impl_getrf!(f32, sgetrf);
impl_getrf!(f64, dgetrf);
impl_getrf!(c32, cgetrf);
impl_getrf!(c64, zgetrf);
