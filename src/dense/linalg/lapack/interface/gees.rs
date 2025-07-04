//! Implementation of ?gees - Compute the eigenvalues and, optionally, the Schur form of a general
//! matrix.

use lapack::{cgees, dgees, sgees, zgees};

use itertools::izip;

use num::{complex::ComplexFloat, Zero};

use crate::base_types::{c32, c64, LapackError};
use crate::{base_types::LapackResult, traits::rlst_num::RlstScalar};

use crate::dense::linalg::lapack::interface::lapack_return;

/// The job options for the Schur vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum JobVs {
    /// Do not compute the Schur vectors.
    None = b'N',
    /// Compute the Schur vectors.
    Compute = b'V',
}

/// ?gees - Compute the eigenvalues and, optionally, the Schur form of a general matrix.
pub trait Gees: RlstScalar {
    /// Compute the eigenvalues and, optionally, the Schur form of a general matrix.
    fn gees(
        jobvs: JobVs,
        n: usize,
        a: &mut [Self],
        lda: usize,
        w: &mut [Self::Complex],
        vs: Option<&mut [Self]>,
        ldvs: usize,
    ) -> LapackResult<()>;
}

macro_rules! implement_gees {
    ($scalar:ty, $gees:expr) => {
        impl Gees for $scalar {
            fn gees(
                jobvs: JobVs,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Complex],
                vs: Option<&mut [Self]>,
                ldvs: usize,
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

                let mut vs_temp = Vec::<$scalar>::new();

                let vs = match jobvs {
                    JobVs::Compute => {
                        let vs = vs.expect("If `jobvs` is Compute, `vs` must be Some.");
                        assert!(
                            ldvs >= std::cmp::max(1, n),
                            "Require `ldvs` {} >= `max(1, n)` {}.",
                            ldvs,
                            std::cmp::max(1, n)
                        );
                        assert_eq!(
                            vs.len(),
                            ldvs * n,
                            "Require `vs.len()` {} == `ldvs * n` {}.",
                            vs.len(),
                            ldvs * n
                        );
                        vs
                    }
                    JobVs::None => {
                        assert!(ldvs >= 1, "Require `ldvs` {} >= 1.", ldvs);
                        vs_temp.as_mut_slice()
                    }
                };

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];

                let mut wr = vec![<$scalar>::zero(); n];
                let mut wi = vec![<$scalar>::zero(); n];
                let mut bwork = Vec::<i32>::new();

                let mut sdim = 0;

                unsafe {
                    $gees(
                        jobvs as u8,
                        b'N',
                        None,
                        n as i32,
                        a,
                        lda as i32,
                        &mut sdim,
                        &mut wr,
                        &mut wi,
                        vs,
                        ldvs as i32,
                        &mut work,
                        -1,
                        &mut bwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];
                unsafe {
                    $gees(
                        jobvs as u8,
                        b'N',
                        None,
                        n as i32,
                        a,
                        lda as i32,
                        &mut sdim,
                        &mut wr,
                        &mut wi,
                        vs,
                        ldvs as i32,
                        &mut work,
                        lwork as i32,
                        &mut bwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                // Fill the output complex eigenvalues
                izip!(w.iter_mut(), wr.iter(), wi.iter()).for_each(|(w, &r, &i)| {
                    *w = <<$scalar as RlstScalar>::Complex>::new(r, i);
                });

                lapack_return(info, ())
            }
        }
    };
}

macro_rules! implement_gees_complex {
    ($scalar:ty, $gees:expr) => {
        impl Gees for $scalar {
            fn gees(
                jobvs: JobVs,
                n: usize,
                a: &mut [Self],
                lda: usize,
                w: &mut [Self::Complex],
                vs: Option<&mut [Self]>,
                ldvs: usize,
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

                let mut vs_temp = Vec::<$scalar>::new();

                let vs = match jobvs {
                    JobVs::Compute => {
                        let vs = vs.expect("If `jobvs` is Compute, `vs` must be Some.");
                        assert!(
                            ldvs >= std::cmp::max(1, n),
                            "Require `ldvs` {} >= `max(1, n)` {}.",
                            ldvs,
                            std::cmp::max(1, n)
                        );
                        assert_eq!(
                            vs.len(),
                            ldvs * n,
                            "Require `vs.len()` {} == `ldvs * n` {}.",
                            vs.len(),
                            ldvs * n
                        );
                        vs
                    }
                    JobVs::None => {
                        assert!(ldvs >= 1, "Require `ldvs` {} >= 1.", ldvs);
                        vs_temp.as_mut_slice()
                    }
                };

                let mut info = 0;
                let mut work = vec![<$scalar>::zero(); 1];

                let mut bwork = Vec::<i32>::new();

                let mut sdim = 0;

                let mut rwork = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); n];

                unsafe {
                    $gees(
                        jobvs as u8,
                        b'N',
                        None,
                        n as i32,
                        a,
                        lda as i32,
                        &mut sdim,
                        w,
                        vs,
                        ldvs as i32,
                        &mut work,
                        -1,
                        &mut rwork,
                        &mut bwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }
                let lwork = work[0].re() as usize;
                let mut work = vec![<$scalar>::zero(); lwork];
                unsafe {
                    $gees(
                        jobvs as u8,
                        b'N',
                        None,
                        n as i32,
                        a,
                        lda as i32,
                        &mut sdim,
                        w,
                        vs,
                        ldvs as i32,
                        &mut work,
                        lwork as i32,
                        &mut rwork,
                        &mut bwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(LapackError::LapackInfoCode(info));
                }

                lapack_return(info, ())
            }
        }
    };
}

implement_gees!(f32, sgees);
implement_gees!(f64, dgees);
implement_gees_complex!(c32, cgees);
implement_gees_complex!(c64, zgees);
