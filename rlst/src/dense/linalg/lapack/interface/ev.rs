//! Implementation of ???ev - Compute the eigenvalues and, optionally eigenvectors of a real
//! symmetric or hermitian matrix.

use lapack::{cheev, dsyev, ssyev, zheev};

use crate::dense::linalg::lapack::interface::lapack_return;

use crate::base_types::{LapackError, LapackResult, c32, c64};
use crate::traits::rlst_num::RlstScalar;

use num::{Zero, complex::ComplexFloat};

/// The job options for the eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JobZEv {
    /// Do not compute the eigenvectors.
    None = b'N',
    /// Compute the eigenvectors.
    Compute = b'V',
}

/// Use upper or lower triangular part of the matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum EvUplo {
    /// Use the upper triangular part of the matrix.
    Upper = b'U',
    /// Use the lower triangular part of the matrix.
    Lower = b'L',
}

/// ???ev - Compute the eigenvalues and, optionally eigenvectors of a real symmetric or hermitian
/// matrix.
pub trait Ev: RlstScalar {
    /// Compute the eigenvalues and, optionally eigenvectors of a real symmetric or hermitian
    /// matrix.
    /// **Arguments:**
    /// - `jobz`: Specifies whether to compute the eigenvectors.
    /// - `uplo`: Specifies whether to use the upper or lower triangular part of the matrix.
    /// - `n`: The order of the matrix A.
    /// - `a`: The matrix A to be factored.
    /// - `lda`: The leading dimension of the matrix A.
    /// - `w`: The vector to store the eigenvalues.
    fn ev(
        jobz: JobZEv,
        uplo: EvUplo,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self::Real],
    ) -> LapackResult<()>;
}

macro_rules! implement_ev {
    ($scalar:ty, $ev:expr) => {
        impl Ev for $scalar {
            fn ev(
                jobz: JobZEv,
                uplo: EvUplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Real],
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

                assert_eq!(w.len(), n, "Require `w.len()` {} == `n` {}.", w.len(), n);

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $ev(
                        jobz as u8, uplo as u8, n as i32, a, lda as i32, w, &mut work, -1,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $ev(
                        jobz as u8,
                        uplo as u8,
                        n as i32,
                        a,
                        lda as i32,
                        w,
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

macro_rules! implement_ev_complex {
    ($scalar:ty, $ev:expr) => {
        impl Ev for $scalar {
            fn ev(
                jobz: JobZEv,
                uplo: EvUplo,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Real],
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

                assert_eq!(w.len(), n, "Require `w.len()` {} == `n` {}.", w.len(), n);

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];
                let mut rwork = vec![
                    <<$scalar as RlstScalar>::Real as Zero>::zero();
                    std::cmp::max(1, 3 * n - 2)
                ];

                unsafe {
                    $ev(
                        jobz as u8, uplo as u8, n as i32, a, lda as i32, w, &mut work, -1,
                        &mut rwork, &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $ev(
                        jobz as u8,
                        uplo as u8,
                        n as i32,
                        a,
                        lda as i32,
                        w,
                        &mut work,
                        lwork as i32,
                        &mut rwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_ev!(f32, ssyev);
implement_ev!(f64, dsyev);
implement_ev_complex!(c32, cheev);
implement_ev_complex!(c64, zheev);
