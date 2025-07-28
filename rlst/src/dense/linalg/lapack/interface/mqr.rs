//! Implementation of ???mqr - Apply an orthogonal or unitary matrix given through elementary
//! reflectors.
//!
//! Note: This trait implements the functionality of ?ormqr for real matrices and ?unmqr for
//! complex matrices.

use lapack::{cunmqr, dormqr, sormqr, zunmqr};

use crate::base_types::LapackResult;
use crate::base_types::{c32, c64, LapackError};

use crate::dense::linalg::lapack::interface::lapack_return;

use num::{complex::ComplexFloat, Zero};

/// Transpose modes for the `?getrs` function.
///
/// Note: For real matrices Q^H is the same as Q^T.
#[derive(Clone, Copy)]
pub enum MqrTransMode {
    ///  Apply Q
    NoTranspose,
    /// Apply Q^H
    ConjugateTranspose,
}

/// Apply Q or Q^H from the left or right.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum MqrSide {
    /// Apply Q/Q^H from the left
    Left = b'L',
    /// Apply Q/Q^H from the right
    Right = b'R',
}

trait TransposeToChar {
    fn to_char(trans: MqrTransMode) -> u8;
}

macro_rules! implement_transpose_to_char {
    ($scalar:ty, $char:expr) => {
        impl TransposeToChar for $scalar {
            fn to_char(trans: MqrTransMode) -> u8 {
                match trans {
                    MqrTransMode::NoTranspose => b'N',
                    MqrTransMode::ConjugateTranspose => $char,
                }
            }
        }
    };
}

implement_transpose_to_char!(f32, b'T');
implement_transpose_to_char!(f64, b'T');
implement_transpose_to_char!(c32, b'C');
implement_transpose_to_char!(c64, b'C');

/// ???mqr - Apply an orthogonal or unitary matrix given through elementary reflectors.
///
/// **Arguments:**
///
/// - `side`: Specifies whether to apply Q from the left or right.
/// - `trans`: Specifies whether to apply Q or Q^H.
/// - `m`: Number of rows of the matrix C.
/// - `n`: Number of columns of the matrix C.
/// - `k`: Number of elementary reflectors.
/// - `a`: The matrix containing the elementary reflectors.
/// - `lda`: Leading dimension of the matrix A.
/// - `tau`: The scalar factors of the elementary reflectors.
/// - `c`: The matrix to which the orthogonal/unitary matrix is applied.
/// - `ldc`: Leading dimension of the matrix c.
///
/// **Returns:**
/// - The Lapack info code if the function fails.
pub trait Mqr: Sized {
    /// Apply an orthogonal or unitary matrix given through elementary reflectors.
    #[allow(clippy::too_many_arguments)]
    fn mqr(
        side: MqrSide,
        trans: MqrTransMode,
        m: usize,
        n: usize,
        k: usize,
        a: &[Self],
        lda: usize,
        tau: &[Self],
        c: &mut [Self],
        ldc: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_mqr {
    ($scalar:ty, $mqr:expr) => {
        impl Mqr for $scalar {
            fn mqr(
                side: MqrSide,
                trans: MqrTransMode,
                m: usize,
                n: usize,
                k: usize,
                a: &[Self],
                lda: usize,
                tau: &[Self],
                c: &mut [Self],
                ldc: usize,
            ) -> LapackResult<()> {
                let mut info = 0;

                match side {
                    MqrSide::Left => {
                        assert!(m >= k, "Require `m` {} >= `k` {}.", m, k);
                        assert!(
                            lda >= std::cmp::max(1, m),
                            "Require `lda` {} >= `max(1, m)` {}.",
                            lda,
                            std::cmp::max(1, m)
                        );
                    }
                    MqrSide::Right => {
                        assert!(n >= k, "Require `n` {} >= `k` {}.", m, k);
                        assert!(
                            lda >= std::cmp::max(1, n),
                            "Require `lda` {} >= `max(1, n)` {}.",
                            lda,
                            std::cmp::max(1, n)
                        );
                    }
                }
                assert_eq!(
                    a.len(),
                    lda * k,
                    "Require `a.len()` {} == `lda * k` {}.",
                    a.len(),
                    lda * k
                );
                assert_eq!(
                    tau.len(),
                    k,
                    "Require `tau.len()` {} == `k` {}.",
                    tau.len(),
                    k
                );

                assert!(
                    ldc >= std::cmp::max(1, m),
                    "Require `ldc` {} >= `max(1, m)` {}.",
                    ldc,
                    std::cmp::max(1, m)
                );

                assert_eq!(
                    c.len(),
                    ldc * n,
                    "Require `c.len()` {} == `ldc * n` {}.",
                    c.len(),
                    ldc * n
                );

                let trans = <$scalar as TransposeToChar>::to_char(trans);
                let mut work = [<$scalar>::zero(); 1];

                unsafe {
                    $mqr(
                        side as u8, trans, m as i32, n as i32, k as i32, a, lda as i32, tau, c,
                        ldc as i32, &mut work, -1, &mut info,
                    );
                }

                let lwork = work[0].re() as i32;

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let mut work = vec![<$scalar>::zero(); lwork as usize];
                unsafe {
                    $mqr(
                        side as u8, trans, m as i32, n as i32, k as i32, a, lda as i32, tau, c,
                        ldc as i32, &mut work, lwork, &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_mqr!(f32, sormqr);
implement_mqr!(f64, dormqr);
implement_mqr!(c32, cunmqr);
implement_mqr!(c64, zunmqr);
