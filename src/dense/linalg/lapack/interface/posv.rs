//! Implementation of ?posv - Solve a positve definite system of linear equations with Cholesky
//! factorization.

use lapack::{cposv, dposv, sposv, zposv};

use crate::dense::linalg::lapack::interface::lapack_return;

use super::{c32, c64, LapackResult};

/// `Uplo` parameter for `?posv` to specify which triangular part of the matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PosvUplo {
    /// Upper triangular part of the matrix is stored.
    Upper = b'U',
    /// Lower triangular part of the matrix is stored.
    Lower = b'L',
}

/// ?posv - Solve a positive definite system of linear equations with Cholesky factorization.
pub trait Posv: Sized {
    /// Solve a positive definite system of linear equations with Cholesky factorization.
    ///
    /// **Arguments:**
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    /// - `n`: The order of the matrix A.
    /// - `nrhs`: The number of right-hand sides (columns in B).
    /// - `a`: The matrix A to be factored.
    /// - `lda`: The leading dimension of the matrix A.
    /// - `b`: The right-hand side matrix B.
    /// - `ldb`: The leading dimension of the matrix B.
    ///
    /// **Returns:**
    /// A `LapackResult<()>` indicating success or failure.
    fn posv(
        uplo: PosvUplo,
        n: usize,
        nrhs: usize,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_posv {
    ($scalar:ty, $posv:expr) => {
        impl Posv for $scalar {
            fn posv(
                uplo: PosvUplo,
                n: usize,
                nrhs: usize,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
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

                assert_eq!(
                    b.len(),
                    ldb * nrhs,
                    "Require `b.len()` {} == `ldb * nrhs` {}.",
                    b.len(),
                    ldb * nrhs
                );

                assert!(
                    ldb >= std::cmp::max(1, n),
                    "Require `ldb` {} >= `max(1, n)` {}.",
                    ldb,
                    std::cmp::max(1, n)
                );

                let mut info = 0;

                unsafe {
                    $posv(
                        uplo as u8,
                        n as i32,
                        nrhs as i32,
                        a,
                        lda as i32,
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

implement_posv!(f32, sposv);
implement_posv!(f64, dposv);
implement_posv!(c32, cposv);
implement_posv!(c64, zposv);
