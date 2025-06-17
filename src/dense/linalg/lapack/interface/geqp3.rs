//! Implementation of ?geqp3 - Column pivoted QR factorization.

use lapack::{cgeqp3, dgeqp3, sgeqp3, zgeqp3};

use crate::dense::linalg::lapack::interface::lapack_return;

use super::{c32, c64, LapackError, LapackResult};

use num::{complex::ComplexFloat, Zero};

/// ?geqp3 - Column pivoted QR factorization.
pub trait Geqp3: Sized {
    /// Perform column pivoted QR factorization of a matrix `a` with dimensions `m` x `n`.
    fn geqp3(
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        jpvt: &mut Vec<i32>,
        tau: &mut Vec<Self>,
    ) -> LapackResult<()>;
}

macro_rules! implement_geqp3 {
    ($scalar:ty, $geqp3:expr) => {
        impl Geqp3 for $scalar {
            fn geqp3(
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                jpvt: &mut Vec<i32>,
                tau: &mut Vec<Self>,
            ) -> LapackResult<()> {
                assert_eq!(
                    lda * n,
                    a.len(),
                    "Require `lda * n` {} == `a.len()` {}.",
                    lda * n,
                    a.len()
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );
                let k = std::cmp::min(m, n);

                assert_eq!(
                    jpvt.len(),
                    n,
                    "Require `jpvt.len()` {} == `n` {}.",
                    jpvt.len(),
                    n
                );

                assert_eq!(
                    tau.len(),
                    k,
                    "Require `tau.len()` {} == `min(m, n)` {}.",
                    tau.len(),
                    k
                );

                let mut info = 0;

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $geqp3(
                        m as i32, n as i32, a, lda as i32, jpvt, tau, &mut work, -1, &mut info,
                    );
                }

                if info != 0 {
                    Err(LapackError::LapackInfoCode(info))
                } else {
                    let lwork = work[0].re() as usize;
                    let mut work = vec![<$scalar>::zero(); lwork];

                    unsafe {
                        $geqp3(
                            m as i32,
                            n as i32,
                            a,
                            lda as i32,
                            jpvt,
                            tau,
                            &mut work,
                            lwork as i32,
                            &mut info,
                        );
                    }
                    lapack_return(info, ())
                }
            }
        }
    };

    ($scalar:ty, $real_scalar:ty, $geqp3:expr) => {
        impl Geqp3 for $scalar {
            fn geqp3(
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
                jpvt: &mut Vec<i32>,
                tau: &mut Vec<Self>,
            ) -> LapackResult<()> {
                assert_eq!(
                    lda * n,
                    a.len(),
                    "Require `lda * n` {} == `a.len()` {}.",
                    lda * n,
                    a.len()
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );
                let k = std::cmp::min(m, n);

                assert_eq!(
                    jpvt.len(),
                    n,
                    "Require `jpvt.len()` {} == `n` {}.",
                    jpvt.len(),
                    n
                );

                assert_eq!(
                    tau.len(),
                    k,
                    "Require `tau.len()` {} == `min(m, n)` {}.",
                    tau.len(),
                    k
                );

                let mut info = 0;

                let mut work = vec![<$scalar>::zero(); 1];

                let mut rwork = vec![<$real_scalar>::zero(); 2 * n];

                unsafe {
                    $geqp3(
                        m as i32, n as i32, a, lda as i32, jpvt, tau, &mut work, -1, &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(LapackError::LapackInfoCode(info))
                } else {
                    let lwork = work[0].re() as usize;
                    let mut work = vec![<$scalar>::zero(); lwork];

                    unsafe {
                        $geqp3(
                            m as i32,
                            n as i32,
                            a,
                            lda as i32,
                            jpvt,
                            tau,
                            &mut work,
                            lwork as i32,
                            &mut rwork,
                            &mut info,
                        );
                    }
                    lapack_return(info, ())
                }
            }
        }
    };
}

implement_geqp3!(f32, sgeqp3);
implement_geqp3!(f64, dgeqp3);
implement_geqp3!(c32, f32, cgeqp3);
implement_geqp3!(c64, f64, zgeqp3);
