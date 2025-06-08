//! Implementation of ?geqrf - QR factorization.

use lapack::{cgeqrf, dgeqrf, sgeqrf, zgeqrf};

use crate::dense::linalg::lapack::interface::lapack_return;

use super::{c32, c64, LapackError, LapackResult};

use num::{complex::ComplexFloat, Zero};

/// ?geqrf - QR factorization.
pub trait Geqrf: Sized {
    /// Perform QR factorization of a matrix `a` with dimensions `m` x `n`.
    ///
    /// Returns a vector of elementary reflectors `tau`.:
    fn geqp3(m: usize, n: usize, a: &mut [Self], lda: usize) -> LapackResult<Vec<Self>>;
}

macro_rules! implement_geqrf {
    ($scalar:ty, $geqrf:expr) => {
        impl Geqrf for $scalar {
            fn geqp3(m: usize, n: usize, a: &mut [Self], lda: usize) -> LapackResult<Vec<Self>> {
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

                let mut tau = vec![<$scalar>::zero(); k];

                let mut info = 0;

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $geqrf(
                        m as i32, n as i32, a, lda as i32, &mut tau, &mut work, -1, &mut info,
                    );
                }

                if info != 0 {
                    Err(LapackError::LapackInfoCode(info))
                } else {
                    let lwork = work[0].re() as usize;
                    let mut work = vec![<$scalar>::zero(); lwork];

                    unsafe {
                        $geqrf(
                            m as i32,
                            n as i32,
                            a,
                            lda as i32,
                            &mut tau,
                            &mut work,
                            lwork as i32,
                            &mut info,
                        );
                    }
                    lapack_return(info, tau)
                }
            }
        }
    };
}

implement_geqrf!(f32, sgeqrf);
implement_geqrf!(f64, dgeqrf);
implement_geqrf!(c32, cgeqrf);
implement_geqrf!(c64, zgeqrf);
