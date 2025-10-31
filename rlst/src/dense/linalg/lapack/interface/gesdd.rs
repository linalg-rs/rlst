//! Implementation of ?gesdd - SVD factorization via divide-and-conquer

use lapack::{cgesdd, dgesdd, sgesdd, zgesdd};

use crate::base_types::{LapackError, c32, c64};
use crate::{base_types::LapackResult, traits::rlst_num::RlstScalar};

use crate::dense::linalg::lapack::interface::lapack_return;

use num::{Zero, complex::ComplexFloat};

/// JobZ specifies the computation of the left and right singular vectors.
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum JobZ {
    /// Return all singular vectors.
    A = b'A',
    /// Return the first `min(m, n)` singular vectors.
    S = b'S',
    /// Do not compute singular vectors.
    N = b'N',
}

/// ?gesdd - SVD factorization via divide-and-conquer
pub trait Gesdd: RlstScalar {
    /// Perform a singular value decomposition (SVD) of a matrix `a` with dimensions `m` x `n`.
    /// If `jobz` is `JobZ::N`, the singular vectors are not computed and `u` and `vt` can be
    /// `None`.
    #[allow(clippy::too_many_arguments)]
    fn gesdd(
        jobz: JobZ,
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

macro_rules! implement_gesdd {
    ($scalar:ty, $gesdd:expr) => {
        impl Gesdd for $scalar {
            fn gesdd(
                jobz: JobZ,
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

                let mut info = 0;

                let k = std::cmp::min(m, n);

                assert_eq!(
                    s.len(),
                    k,
                    "Require `s.len()` {} == `min(m, n)` {}.",
                    s.len(),
                    k
                );

                // These are only needed if the user requests no singular vectors.
                // The reason is that Lapack always requires some reference to a singular vector slice,
                // even if it is not referenced in the computation.
                let mut u_temp = Vec::<$scalar>::new();
                let mut vt_temp = Vec::<$scalar>::new();

                let (u, vt) = match jobz {
                    JobZ::A => {
                        let u = u.expect("JobU::A requires u to be Some");
                        let vt = vt.expect("JobVt::A requires vt to be Some");

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

                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `vt.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );
                        assert!(
                            ldvt >= std::cmp::max(1, n),
                            "Require `ldvt` {} >= `max(1, n)` {}.",
                            ldvt,
                            std::cmp::max(1, n)
                        );

                        (u, vt)
                    }
                    JobZ::S => {
                        let u = u.expect("JobU::S requires u to be Some");
                        let vt = vt.expect("JobVt::S requires vt to be Some");

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

                        (u, vt)
                    }
                    JobZ::N => {
                        let u = &mut u_temp;
                        let vt = &mut vt_temp;
                        assert!(ldu >= 1, "Require `ldu` {} >= 1.", ldu);
                        assert!(ldvt >= 1, "Require `ldvt` {} >= 1.", ldvt);
                        (u.as_mut_slice(), vt.as_mut_slice())
                    }
                };

                let mut iwork = vec![0 as i32; 8 * k];

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $gesdd(
                        jobz as u8,
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
                        &mut iwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as i32;

                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $gesdd(
                        jobz as u8,
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
                        &mut iwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

macro_rules! implement_gesdd_complex {
    ($scalar:ty, $gesdd:expr) => {
        impl Gesdd for $scalar {
            fn gesdd(
                jobz: JobZ,
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

                let mut info = 0;

                let k = std::cmp::min(m, n);

                assert_eq!(
                    s.len(),
                    k,
                    "Require `s.len()` {} == `min(m, n)` {}.",
                    s.len(),
                    k
                );

                // These are only needed if the user requests no singular vectors.
                // The reason is that Lapack always requires some reference to a singular vector slice,
                // even if it is not referenced in the computation.
                let mut u_temp = Vec::<$scalar>::new();
                let mut vt_temp = Vec::<$scalar>::new();

                let (u, vt) = match jobz {
                    JobZ::A => {
                        let u = u.expect("JobU::A requires u to be Some");
                        let vt = vt.expect("JobVt::A requires vt to be Some");

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

                        assert_eq!(
                            vt.len(),
                            ldvt * n,
                            "Require `vt.len()` {} == `ldvt * n` {}.",
                            vt.len(),
                            ldvt * n
                        );
                        assert!(
                            ldvt >= std::cmp::max(1, n),
                            "Require `ldvt` {} >= `max(1, n)` {}.",
                            ldvt,
                            std::cmp::max(1, n)
                        );

                        (u, vt)
                    }
                    JobZ::S => {
                        let u = u.expect("JobU::S requires u to be Some");
                        let vt = vt.expect("JobVt::S requires vt to be Some");

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

                        (u, vt)
                    }
                    JobZ::N => {
                        let u = &mut u_temp;
                        let vt = &mut vt_temp;
                        assert!(ldu >= 1, "Require `ldu` {} >= 1.", ldu);
                        assert!(ldvt >= 1, "Require `ldvt` {} >= 1.", ldvt);
                        (u.as_mut_slice(), vt.as_mut_slice())
                    }
                };

                let mx = std::cmp::max(m, n);
                let mn = std::cmp::min(m, n);
                // The formula for the optimal workspace size is taken from the LAPACK
                // documentation.
                // In old versions of LAPACK, at least 7 * mn is required, but in newer versions
                // it is 5 * mn. The formula below is a compromise that works for both at the cost
                // of a slightly larger workspace.
                let lrwork = std::cmp::max(5 * mn * mn + 7 * mn, 2 * mx * mn + 2 * mn * mn + mn);

                let mut rwork =
                    vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); lrwork as usize];
                let mut iwork = vec![0 as i32; 8 * mn];

                let mut work = vec![<$scalar>::zero(); 1];

                unsafe {
                    $gesdd(
                        jobz as u8,
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
                        &mut iwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as i32;

                let mut work = vec![<$scalar>::zero(); lwork as usize];

                unsafe {
                    $gesdd(
                        jobz as u8,
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
                        &mut iwork,
                        &mut info,
                    );
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_gesdd!(f32, sgesdd);
implement_gesdd!(f64, dgesdd);
implement_gesdd_complex!(c32, cgesdd);
implement_gesdd_complex!(c64, zgesdd);
