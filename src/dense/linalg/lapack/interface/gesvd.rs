//! Implementation of ?gesvd - SVD factorization

use lapack::{cgesvd, dgesvd, sgesvd, zgesvd};

use crate::base_types::{c32, c64, LapackError};
use crate::{base_types::LapackResult, traits::rlst_num::RlstScalar};

use crate::dense::linalg::lapack::interface::lapack_return;

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
    /// If either `jobu` or `jobvt` is `JobU::N` or `JobVt::N`, the corresponding singular vectors
    /// are not computed, and the array u or correspondingly vt is not referenced and can be
    /// `None`.
    fn gesvd(
        jobu: JobU,
        jobvt: JobVt,
        m: usize,
        n: usize,
        a: &mut [Self],
        lda: usize,
        s: &mut [Self::Real],
        u: Option<&mut [Self]>,
        ldu: usize,
        vt: Option<&mut [Self]>,
        ldvt: usize,
    ) -> LapackResult<()>;
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
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: usize,
                vt: Option<&mut [Self]>,
                ldvt: usize,
            ) -> LapackResult<()> {
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

                let k = std::cmp::min(m, n);

                assert_eq!(
                    s.len(),
                    k,
                    "Require `s.len()` {} == `min(m, n)` {}.",
                    s.len(),
                    k,
                );

                let mut info = 0;

                let mut u_temp = Vec::<$scalar>::new();
                let mut vt_temp = Vec::<$scalar>::new();

                let u = match jobu {
                    JobU::A => {
                        let u = u.expect("JobU::A requires u to be Some");
                        assert_eq!(
                            u.len(),
                            ldu * m,
                            "Require `u.len()` {} == `ldu * m` {}.",
                            u.len(),
                            ldu * m
                        );

                        assert!(
                            ldu >= std::cmp::max(1, m),
                            "Require `ldu` {} >= `max(1, m)` {}.",
                            ldu,
                            std::cmp::max(1, m)
                        );
                        u
                    }
                    JobU::S => {
                        let u = u.expect("JobU::S requires u to be Some");
                        assert_eq!(
                            u.len(),
                            ldu * k,
                            "Require `u.len()` {} == `ldu * min(m, n)` {}.",
                            u.len(),
                            ldu * k
                        );
                        assert!(
                            ldu >= std::cmp::max(1, m),
                            "Require `ldu` {} >= `max(1, m)` {}.",
                            ldu,
                            std::cmp::max(1, m)
                        );
                        u
                    }
                    JobU::N => {
                        assert!(ldu >= 1, "Require `ldu` {} >= 1.", ldu);
                        u_temp.as_mut_slice()
                    }
                };

                let vt = match jobvt {
                    JobVt::A => {
                        let vt = vt.expect("JobVt::A requires vt to be Some");
                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `u.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );

                        assert!(
                            ldvt >= std::cmp::max(1, n),
                            "Require `ldvt` {} >= `max(1, n)` {}.",
                            ldvt,
                            std::cmp::max(1, k)
                        );
                        vt
                    }
                    JobVt::S => {
                        let vt = vt.expect("JobVt::S requires u to be Some");
                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `vt.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );
                        assert!(
                            ldvt >= std::cmp::max(1, k),
                            "Require `ldvt` {} >= `max(1, min(m, n))` {}.",
                            ldvt,
                            std::cmp::max(1, k)
                        );
                        vt
                    }
                    JobVt::N => {
                        assert!(ldvt >= 1, "Require `ldvt` {} >= 1.", ldvt);
                        vt_temp.as_mut_slice()
                    }
                };

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        s,
                        u,
                        ldu as i32,
                        vt,
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
                        s,
                        u,
                        ldu as i32,
                        vt,
                        ldvt as i32,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
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
                s: &mut [Self::Real],
                u: Option<&mut [Self]>,
                ldu: usize,
                vt: Option<&mut [Self]>,
                ldvt: usize,
            ) -> LapackResult<()> {
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

                let k = std::cmp::min(m, n);

                assert_eq!(
                    s.len(),
                    k,
                    "Require `s.len()` {} == `min(m, n)` {}.",
                    s.len(),
                    k,
                );

                let mut info = 0;

                let mut rwork = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 5 * k];

                let mut u_temp = Vec::<$scalar>::new();
                let mut vt_temp = Vec::<$scalar>::new();

                let u = match jobu {
                    JobU::A => {
                        let u = u.expect("JobU::A requires u to be Some");
                        assert_eq!(
                            u.len(),
                            ldu * m,
                            "Require `u.len()` {} == `ldu * m` {}.",
                            u.len(),
                            ldu * m
                        );

                        assert!(
                            ldu >= std::cmp::max(1, m),
                            "Require `ldu` {} >= `max(1, m)` {}.",
                            ldu,
                            std::cmp::max(1, m)
                        );
                        u
                    }
                    JobU::S => {
                        let u = u.expect("JobU::S requires u to be Some");
                        assert_eq!(
                            u.len(),
                            ldu * k,
                            "Require `u.len()` {} == `ldu * min(m, n)` {}.",
                            u.len(),
                            ldu * k
                        );
                        assert!(
                            ldu >= std::cmp::max(1, m),
                            "Require `ldu` {} >= `max(1, m)` {}.",
                            ldu,
                            std::cmp::max(1, m)
                        );
                        u
                    }
                    JobU::N => {
                        assert!(ldu >= 1, "Require `ldu` {} >= 1.", ldu);
                        u_temp.as_mut_slice()
                    }
                };

                let vt = match jobvt {
                    JobVt::A => {
                        let vt = vt.expect("JobVt::A requires vt to be Some");
                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `u.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );

                        assert!(
                            ldvt >= std::cmp::max(1, n),
                            "Require `ldvt` {} >= `max(1, n)` {}.",
                            ldvt,
                            std::cmp::max(1, k)
                        );
                        vt
                    }
                    JobVt::S => {
                        let vt = vt.expect("JobVt::S requires u to be Some");
                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `vt.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );
                        assert!(
                            ldvt >= std::cmp::max(1, k),
                            "Require `ldvt` {} >= `max(1, min(m, n))` {}.",
                            ldvt,
                            std::cmp::max(1, k)
                        );
                        vt
                    }
                    JobVt::N => {
                        assert!(ldvt >= 1, "Require `ldvt` {} >= 1.", ldvt);
                        vt_temp.as_mut_slice()
                    }
                };

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $gesvd(
                        jobu as u8,
                        jobvt as u8,
                        m as i32,
                        n as i32,
                        a,
                        lda as i32,
                        s,
                        u,
                        ldu as i32,
                        vt,
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
                        s,
                        u,
                        ldu as i32,
                        vt,
                        ldvt as i32,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_gesvd!(f32, sgesvd);
implement_gesvd!(f64, dgesvd);
implement_gesvd_complex!(c32, cgesvd);
implement_gesvd_complex!(c64, zgesvd);
