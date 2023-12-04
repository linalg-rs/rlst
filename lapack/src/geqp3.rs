//! Interface to dgeqp3

use lapack::{cgeqp3, dgeqp3, sgeqp3, zgeqp3};
use num::Zero;
use rlst_common::types::*;

pub trait Geqp3: Scalar {
    fn geqp3(m: i32, n: i32, a: &mut [Self], lda: i32, jpvt: &mut [i32], tau: &mut [Self]) -> i32;
}

macro_rules! impl_geqp3_real {
    ($scalar:ty, $geqp3:expr) => {
        impl Geqp3 for $scalar {
            fn geqp3(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
            ) -> i32 {
                assert!(m >= 0);
                assert!(n >= 0);
                assert!(lda >= 0);
                assert_eq!(a.len() as i32, lda * n);
                assert_eq!(jpvt.len() as i32, n);
                assert_eq!(tau.len() as i32, std::cmp::min(m, n));

                let mut info = 0;
                let lwork = -1;
                let mut work_query = [<Self as Zero>::zero()];
                unsafe { $geqp3(m, n, a, lda, jpvt, tau, &mut work_query, lwork, &mut info) };
                assert_eq!(info, 0);
                let lwork = work_query[0].re() as i32;

                let mut work = vec![<Self as Zero>::zero(); lwork as usize];
                unsafe { $geqp3(m, n, a, lda, jpvt, tau, &mut work, lwork, &mut info) };
                info
            }
        }
    };
}

macro_rules! impl_geqp3_complex {
    ($scalar:ty, $geqp3:expr) => {
        impl Geqp3 for $scalar {
            fn geqp3(
                m: i32,
                n: i32,
                a: &mut [Self],
                lda: i32,
                jpvt: &mut [i32],
                tau: &mut [Self],
            ) -> i32 {
                assert!(m >= 0);
                assert!(n >= 0);
                assert!(lda >= 0);
                assert_eq!(a.len() as i32, lda * n);
                assert_eq!(jpvt.len() as i32, n);
                assert_eq!(tau.len() as i32, std::cmp::min(m, n));
                let mut rwork = vec![<<Self as Scalar>::Real as Zero>::zero(); 2 * n as usize];

                let mut info = 0;
                let lwork = -1;
                let mut work_query = [<Self as Zero>::zero()];
                unsafe {
                    $geqp3(
                        m,
                        n,
                        a,
                        lda,
                        jpvt,
                        tau,
                        &mut work_query,
                        lwork,
                        &mut rwork,
                        &mut info,
                    )
                };
                assert_eq!(info, 0);
                let lwork = work_query[0].re() as i32;

                let mut work = vec![<Self as Zero>::zero(); lwork as usize];
                unsafe {
                    $geqp3(
                        m, n, a, lda, jpvt, tau, &mut work, lwork, &mut rwork, &mut info,
                    )
                };
                info
            }
        }
    };
}

impl_geqp3_real!(f32, sgeqp3);
impl_geqp3_real!(f64, dgeqp3);
impl_geqp3_complex!(c32, cgeqp3);
impl_geqp3_complex!(c64, zgeqp3);
