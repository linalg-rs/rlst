//! Implementation of ?trsm - Solving triangular systems of linear equations.

use blas::{ctrsm, dtrsm, strsm, ztrsm};

use crate::dense::linalg::lapack::interface::lapack_return;

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64};

/// Left: Solve AX = B. Right: Solve XA = B.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrsmSide {
    /// Solve AX = B
    Left = b'L',
    /// Solve XA = B
    Right = b'R',
}

/// Uplo: Specifies whether the upper or lower triangular part of the matrix is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrsmUplo {
    /// Upper triangular part of the matrix is stored.
    Upper = b'U',
    /// Lower triangular part of the matrix is stored.
    Lower = b'L',
}

/// Transpose: Specifies whether the matrix is transposed or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrsmTransA {
    /// No transpose: A is used as is.
    NoTrans = b'N',
    /// Transpose
    Transpose = b'T',
    /// Conjugate transpose
    ConjugateTranspose = b'C',
}

/// Diagonal: Specifies whether the matrix is unit diagonal or not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrsmDiag {
    /// Non-unit diagonal
    NonUnit = b'N',
    /// Unit diagonal
    Unit = b'U',
}

/// ?trsm - Solve triangular systems of linear equations.
pub trait Trsm {
    /// Solve a triangular system of linear equations.
    ///
    /// **Arguments:**
    /// - `side`: Specifies whether to solve AX = B (Left) or XA = B (Right).
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    /// - `transa`: Specifies whether the matrix is transposed or not.
    /// - `diag`: Specifies whether the matrix is unit diagonal or not.
    /// - `m`: The number of rows of the matrix B.
    /// - `n`: The number of columns of the matrix B.
    /// - `alpha`: Scalar multiplier for B.
    /// - `a`: The triangular matrix A.
    /// - `lda`: The leading dimension of the matrix A.
    /// - `b`: The right-hand side matrix B. On exit contains the solution.
    /// - `ldb`: The leading dimension of the matrix B.
    #[allow(clippy::too_many_arguments)]
    fn trsm(
        side: TrsmSide,
        uplo: TrsmUplo,
        transa: TrsmTransA,
        diag: TrsmDiag,
        m: usize,
        n: usize,
        alpha: Self,
        a: &mut [Self],
        lda: usize,
        b: &mut [Self],
        ldb: usize,
    ) -> LapackResult<()>
    where
        Self: Sized;
}

macro_rules! implement_trsm {
    ($scalar:ty, $trsm:expr) => {
        impl Trsm for $scalar {
            fn trsm(
                side: TrsmSide,
                uplo: TrsmUplo,
                transa: TrsmTransA,
                diag: TrsmDiag,
                m: usize,
                n: usize,
                alpha: Self,
                a: &mut [Self],
                lda: usize,
                b: &mut [Self],
                ldb: usize,
            ) -> LapackResult<()> {
                assert_eq!(
                    a.len(),
                    match side {
                        TrsmSide::Left => lda * m,
                        TrsmSide::Right => lda * n,
                    },
                    "Require `a.len()`  == `lda * m`  or `lda * n` depending on the side.",
                );

                assert!(
                    lda >= std::cmp::max(
                        1,
                        match side {
                            TrsmSide::Left => m,
                            TrsmSide::Right => n,
                        }
                    ),
                    "Require `lda` {} >= `max(1, {})` {}.",
                    lda,
                    match side {
                        TrsmSide::Left => m,
                        TrsmSide::Right => n,
                    },
                    std::cmp::max(
                        1,
                        match side {
                            TrsmSide::Left => m,
                            TrsmSide::Right => n,
                        }
                    )
                );

                assert_eq!(
                    b.len(),
                    ldb * n,
                    "Require `b.len()` {} == `ldb * n` {}.",
                    b.len(),
                    ldb * n
                );

                assert!(
                    ldb >= std::cmp::max(1, m),
                    "Require `ldb` {} >= `max(1, m)` {}.",
                    ldb,
                    std::cmp::max(1, m)
                );

                unsafe {
                    $trsm(
                        side as u8,
                        uplo as u8,
                        transa as u8,
                        diag as u8,
                        m as i32,
                        n as i32,
                        alpha,
                        a,
                        lda as i32,
                        b,
                        ldb as i32,
                    );
                }

                lapack_return(0, ())
            }
        }
    };
}

implement_trsm!(f32, strsm);
implement_trsm!(f64, dtrsm);
implement_trsm!(c32, ctrsm);
implement_trsm!(c64, ztrsm);
