//! Implementation of ?geev - Compute the eigenvalues and, optionally, the eigenvectors of a
//! general matrix.

use lapack::{cgeev, dgeev, sgeev, zgeev};

use crate::{dense::linalg::lapack::interface::lapack_return, RlstScalar};

use itertools::izip;

use super::{c32, c64, LapackError, LapackResult};

use num::{complex::ComplexFloat, Zero};

/// The job options for the left eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JobVl {
    /// Do not compute the left eigenvectors.
    None = b'N',
    /// Compute the left eigenvectors.
    Compute = b'V',
}

/// The job options for the right eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JobVr {
    /// Do not compute the right eigenvectors.
    None = b'N',
    /// Compute the right eigenvectors.
    Compute = b'V',
}

/// ?geev - Compute the eigenvalues and, optionally, the eigenvectors of a general matrix.
pub trait Geev: RlstScalar {
    /// Compute the eigenvalues and, optionally, the eigenvectors of a general matrix.
    fn geev(
        jobvl: JobVl,
        jobvr: JobVr,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self::Complex],
        vl: Option<&mut [Self]>,
        ldvl: usize,
        vr: Option<&mut [Self]>,
        ldvr: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_geev {
    ($scalar:ty, $geev:expr) => {
        impl Geev for $scalar {
            fn geev(
                jobvl: JobVl,
                jobvr: JobVr,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Complex],
                vl: Option<&mut [Self]>,
                ldvl: usize,
                vr: Option<&mut [Self]>,
                ldvr: usize,
            ) -> LapackResult<()> {
                let mut info = 0;

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

                assert_eq!(w.len(), n, "Require `wr.len()` {} == `n` {}.", w.len(), n);

                let mut wr = vec![<$scalar>::zero(); n];
                let mut wi = vec![<$scalar>::zero(); n];

                let mut vl_temp = Vec::<$scalar>::new();
                let mut vr_temp = Vec::<$scalar>::new();

                let vl = match jobvl {
                    JobVl::Compute => {
                        let vl = vl.expect("vl must be Some when jobvl is V");
                        assert_eq!(
                            vl.len(),
                            ldvl * n,
                            "Require `vl.len()` {} == `ldvl * n` {}.",
                            vl.len(),
                            ldvl * n
                        );
                        vl
                    }
                    JobVl::None => {
                        assert_eq!(ldvl, 1, "Require `ldvl` {} == 1.", ldvl);
                        vl_temp.as_mut_slice()
                    }
                };
                let vr = match jobvr {
                    JobVr::Compute => {
                        let vr = vr.expect("vr must be Some when jobvr is V");
                        assert_eq!(
                            vr.len(),
                            ldvr * n,
                            "Require `vr.len()` {} == `ldvr * n` {}.",
                            vr.len(),
                            ldvr * n
                        );
                        vr
                    }
                    JobVr::None => {
                        assert_eq!(ldvr, 1, "Require `ldvr` {} == 1.", ldvr);
                        vr_temp.as_mut_slice()
                    }
                };

                let mut work = vec![<$scalar>::zero(); 1];
                unsafe {
                    $geev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        &mut wr,
                        &mut wi,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        -1,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;

                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $geev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        &mut wr,
                        &mut wi,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        lwork as i32,
                        &mut info,
                    );
                }

                for (w_elem, wr_elem, wi_elem) in izip!(w.iter_mut(), wr.iter(), wi.iter()) {
                    *w_elem = <<$scalar as RlstScalar>::Complex>::new(*wr_elem, *wi_elem);
                }

                lapack_return(info, ())
            }
        }
    };
}

macro_rules! implement_geev_complex {
    ($scalar:ty, $geev:expr) => {
        impl Geev for $scalar {
            fn geev(
                jobvl: JobVl,
                jobvr: JobVr,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Complex],
                vl: Option<&mut [Self]>,
                ldvl: usize,
                vr: Option<&mut [Self]>,
                ldvr: usize,
            ) -> LapackResult<()> {
                let mut info = 0;

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

                assert_eq!(w.len(), n, "Require `wr.len()` {} == `n` {}.", w.len(), n);

                let mut vl_temp = Vec::<$scalar>::new();
                let mut vr_temp = Vec::<$scalar>::new();

                let vl = match jobvl {
                    JobVl::Compute => {
                        let vl = vl.expect("vl must be Some when jobvl is V");
                        assert_eq!(
                            vl.len(),
                            ldvl * n,
                            "Require `vl.len()` {} == `ldvl * n` {}.",
                            vl.len(),
                            ldvl * n
                        );
                        vl
                    }
                    JobVl::None => {
                        assert_eq!(ldvl, 1, "Require `ldvl` {} == 1.", ldvl);
                        vl_temp.as_mut_slice()
                    }
                };
                let vr = match jobvr {
                    JobVr::Compute => {
                        let vr = vr.expect("vr must be Some when jobvr is V");
                        assert_eq!(
                            vr.len(),
                            ldvr * n,
                            "Require `vr.len()` {} == `ldvr * n` {}.",
                            vr.len(),
                            ldvr * n
                        );
                        vr
                    }
                    JobVr::None => {
                        assert_eq!(ldvr, 1, "Require `ldvr` {} == 1.", ldvr);
                        vr_temp.as_mut_slice()
                    }
                };

                let mut rwork = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 2 * n];

                let mut work = vec![<$scalar>::zero(); 1];
                unsafe {
                    $geev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        w,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
                        &mut work,
                        -1,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                let lwork = work[0].re() as usize;

                let mut work = vec![<$scalar>::zero(); lwork];

                unsafe {
                    $geev(
                        jobvl as u8,
                        jobvr as u8,
                        n as i32,
                        a,
                        lda as i32,
                        w,
                        vl,
                        ldvl as i32,
                        vr,
                        ldvr as i32,
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

implement_geev!(f32, sgeev);
implement_geev!(f64, dgeev);
implement_geev_complex!(c32, cgeev);
implement_geev_complex!(c64, zgeev);
