//! Implementation of ?gesvd - SVD factorization

use lapack::{cgesvd, dgesvd, sgesvd, zgesvd};

use crate::{dense::linalg::lapack::interface::lapack_return, RlstScalar};

use super::{c32, c64, LapackError, LapackResult};

use num::{complex::ComplexFloat, Zero};

/// JobU specifies the computation of the left singular vectors.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum JobU {
    /// Return all columns.
    A = b'A',
    /// Return the first `min(m, n)` columns.
    S = b'S',
    /// Do not compute U.
    N = b'N',
}

/// JobVt specifies the computation of the right singular vectors.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum JobVt {
    /// Return all rows.
    A = b'A',
    /// Return the first `min(m, n)` rows.
    S = b'S',
    /// Do not compute Vt.
    N = b'N',
}

/// ?gesvd - SVD factorization
pub trait Gesvd: RlstScalar {
    /// Perform a singular value decomposition (SVD) of a matrix `a` with dimensions `m` x `n`.
    fn gesvd(
        jobu: JobU,
        jobvt: JobVt,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
    ) -> LapackResult<(Vec<Self::Real>, Option<Vec<Self>>, Option<Vec<Self>>)>;
}

macro_rules! implement_gesvd {
    ($scalar:ty, $gesvd:expr) => {
        impl Gesvd for $scalar {
            fn gesvd(
                jobu: JobU,
                jobvt: JobVt,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> LapackResult<(Vec<Self::Real>, Option<Vec<Self>>, Option<Vec<Self>>)> {
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );

                let mut info = 0;

                let k = std::cmp::min(m, n);

                let (mut u, ldu) = match jobu {
                    JobU::A => (vec![<$scalar>::zero(); m * m], m),
                    JobU::S => (vec![<$scalar>::zero(); m * k], m),
                    // Just a dummy array for when u is not referenced.
                    JobU::N => (vec![<$scalar>::zero(); 1], 1),
                };

                let (mut vt, ldvt) = match jobvt {
                    JobVt::A => (vec![<$scalar>::zero(); n * n], n),
                    JobVt::S => (vec![<$scalar>::zero(); k * n], k),
                    // Just a dummy array for when vt is not referenced.
                    JobVt::N => (vec![<$scalar>::zero(); 1], 1),
                };

                let mut s = vec![<$scalar>::zero(); k];

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        -1,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as i32;

                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                let u = match jobu {
                    JobU::N => None,
                    _ => Some(u),
                };
                let vt = match jobvt {
                    JobVt::N => None,
                    _ => Some(vt),
                };

                lapack_return(info, (s, u, vt))
            }
        }
    };
}

macro_rules! implement_gesvd_complex {
    ($scalar:ty, $gesvd:expr) => {
        impl Gesvd for $scalar {
            fn gesvd(
                jobu: JobU,
                jobvt: JobVt,
                m: usize,
                n: usize,
                a: &mut [Self],
                lda: usize,
            ) -> LapackResult<(Vec<Self::Real>, Option<Vec<Self>>, Option<Vec<Self>>)> {
                assert_eq!(
                    a.len(),
                    lda * n,
                    "Require `a.len()` {} == `lda * n` {}.",
                    a.len(),
                    lda * n
                );

                assert!(
                    lda >= std::cmp::max(1, m),
                    "Require `lda` {} >= `max(1, m)` {}.",
                    lda,
                    std::cmp::max(1, m)
                );

                let mut info = 0;

                let k = std::cmp::min(m, n);

                let (mut u, ldu) = match jobu {
                    JobU::A => (vec![<$scalar>::zero(); m * m], m),
                    JobU::S => (vec![<$scalar>::zero(); m * k], m),
                    // Just a dummy array for when u is not referenced.
                    JobU::N => (vec![<$scalar>::zero(); 1], 1),
                };

                let (mut vt, ldvt) = match jobvt {
                    JobVt::A => (vec![<$scalar>::zero(); n * n], n),
                    JobVt::S => (vec![<$scalar>::zero(); k * n], k),
                    // Just a dummy array for when vt is not referenced.
                    JobVt::N => (vec![<$scalar>::zero(); 1], 1),
                };

                let mut s = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); k];

                let mut work = vec![<$scalar>::zero(); 1];

                let mut rwork =
                    vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 5 * std::cmp::max(m, n)];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as i32;

                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        &mut s,
                        &mut u,
                        ldu as i32,
                        &mut vt,
                        ldvt as i32,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                let u = match jobu {
                    JobU::N => None,
                    _ => Some(u),
                };
                let vt = match jobvt {
                    JobVt::N => None,
                    _ => Some(vt),
                };

                lapack_return(info, (s, u, vt))
            }
        }
    };
}

implement_gesvd!(f32, sgesvd);
implement_gesvd!(f64, dgesvd);
implement_gesvd_complex!(c32, cgesvd);
implement_gesvd_complex!(c64, zgesvd);
