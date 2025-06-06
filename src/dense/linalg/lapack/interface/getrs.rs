//! Implementation of ?getrs - Solve a system of linear equations using LU factorization.

use lapack::{cgetrs, dgetrs, sgetrs, zgetrs};

use crate::dense::linalg::lapack::interface::lapack_return;

use super::{c32, c64, LapackError, LapackResult};

use num::{complex::ComplexFloat, Zero};

/// Transpose modes for the `?getrs` function.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum GetrsTransMode {
    /// No transpose, solve `AX = b`.
    NoTranspose = b'N',
    /// Transpose, solve `A^T X = b`.
    Transpose = b'T',
    /// Conjugate transpose, solve `A^H X = b`.
    ConjugateTranspose = b'C',
}

/// ?getrs - Solve a system of linear equations using LU factorization.
///
/// **Arguments:**
///
///
///
/// **Returns:**
/// - The Lapack info code if the solve fails.
pub trait Getrs: Sized {
    /// Solve a system of linear equations using LU factorization.
    fn getrs(
        trans: GetrsTransMode,
        n: usize,
        nrhs: usize,
        a: &[Self],
        lda: usize,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_getrs {
    ($scalar:ty, $getrs:expr) => {
        impl Getrs for $scalar {
            fn getrs(
                trans: GetrsTransMode,
                n: usize,
                nrhs: usize,
                a: &[Self],
                lda: usize,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
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

                assert!(
                    ldb >= std::cmp::max(1, n),
                    "Require `ldb` {} >= `max(1, N)` {} .",
                    ldb,
                    std::cmp::max(1, n)
                );
                assert_eq!(
                    b.len(),
                    ldb * nrhs,
                    "Require `b.len()` {} == `ldb * nrhs` {}.",
                    b.len(),
                    ldb * nrhs,
                );

                unsafe {
                    $getrs(
                        trans as u8,
                        n as i32,
                        nrhs as i32,
                        a,
                        lda as i32,
                        ipiv,
                        b,
                        ldb as i32,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_getrs!(f32, sgetrs);
implement_getrs!(f64, dgetrs);
implement_getrs!(c32, cgetrs);
implement_getrs!(c64, zgetrs);
