//! Implementation of ormqr

use lapack::cunmqr;
use lapack::dormqr;
use lapack::sormqr;
use lapack::zunmqr;
use num::Zero;
use rlst_common::types::*;

pub trait Ormqr: Scalar {
    fn ormqr(
        side: u8,
        trans: u8,
        m: i32,
        n: i32,
        k: i32,
        a: &mut [Self],
        lda: i32,
        tau: &mut [Self],
        c: &mut [Self],
        ldc: i32,
        work: Option<&mut [Self]>,
    ) -> i32;

    fn ormqr_query_work(side: u8, trans: u8, m: i32, n: i32, k: i32) -> i32;
}

macro_rules! impl_ormqr_complex {
    ($scalar:ty, $ormqr:expr) => {
        impl Ormqr for $scalar {
            fn ormqr(
                side: u8,
                trans: u8,
                m: i32,
                n: i32,
                k: i32,
                a: &mut [Self],
                lda: i32,
                tau: &mut [Self],
                c: &mut [Self],
                ldc: i32,
                work: Option<&mut [Self]>,
            ) -> i32 {
                assert!(side == b'L' || side == b'R');
                assert!(trans == b'C' || trans == b'N');
                assert!(m >= 0);
                assert!(n >= 0);

                assert!(if side == b'L' {
                    k >= 0 && k <= m
                } else {
                    k >= 0 && k <= n
                });
                assert!(if side == b'L' {
                    lda >= std::cmp::max(1, m)
                } else {
                    lda >= std::cmp::max(1, n)
                });
                assert_eq!(a.len() as i32, lda * k);
                assert_eq!(tau.len() as i32, k);
                assert!(ldc >= std::cmp::max(1, m));
                assert_eq!(c.len() as i32, ldc * n);
                let mut my_work = Vec::<Self>::new();
                let work = if let Some(work) = work {
                    assert!(if side == b'L' {
                        work.len() as i32 >= std::cmp::max(1, n)
                    } else {
                        work.len() as i32 >= std::cmp::max(1, m)
                    });
                    work
                } else {
                    let len = <Self as Ormqr>::ormqr_query_work(side, trans, m, n, k) as usize;
                    my_work.resize(len, <Self as Zero>::zero());
                    &mut my_work
                };

                let mut info = 0;
                unsafe {
                    $ormqr(
                        side,
                        trans,
                        m,
                        n,
                        k,
                        a,
                        lda,
                        tau,
                        c,
                        ldc,
                        work,
                        work.len() as i32,
                        &mut info,
                    )
                };
                info
            }

            fn ormqr_query_work(side: u8, trans: u8, m: i32, n: i32, k: i32) -> i32 {
                let a = [<Self as Zero>::zero()];
                let tau = [<Self as Zero>::zero()];
                let mut c = [<Self as Zero>::zero()];
                let mut work_query = [<Self as Zero>::zero()];
                let mut info = 0;

                assert!(side == b'L' || side == b'R');
                assert!(trans == b'T' || trans == b'N');
                assert!(m >= 0);
                assert!(n >= 0);
                assert!(if side == b'L' {
                    k >= 0 && k <= m
                } else {
                    k >= 0 && k <= n
                });
                let lda = if side == b'L' { m } else { n };
                unsafe {
                    $ormqr(
                        side,
                        trans,
                        m,
                        n,
                        k,
                        &a,
                        lda,
                        &tau,
                        &mut c,
                        m,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                assert_eq!(info, 0);
                work_query[0].re() as i32
            }
        }
    };
}

macro_rules! impl_ormqr_real {
    ($scalar:ty, $ormqr:expr) => {
        impl Ormqr for $scalar {
            fn ormqr(
                side: u8,
                trans: u8,
                m: i32,
                n: i32,
                k: i32,
                a: &mut [Self],
                lda: i32,
                tau: &mut [Self],
                c: &mut [Self],
                ldc: i32,
                work: Option<&mut [Self]>,
            ) -> i32 {
                assert!(side == b'L' || side == b'R');
                assert!(trans == b'T' || trans == b'N');
                assert!(m >= 0);
                assert!(n >= 0);

                assert!(if side == b'L' {
                    k >= 0 && k <= m
                } else {
                    k >= 0 && k <= n
                });
                assert!(if side == b'L' {
                    lda >= std::cmp::max(1, m)
                } else {
                    lda >= std::cmp::max(1, n)
                });
                assert_eq!(a.len() as i32, lda * k);
                assert_eq!(tau.len() as i32, k);
                assert!(ldc >= std::cmp::max(1, m));
                assert_eq!(c.len() as i32, ldc * n);
                let mut my_work = Vec::<Self>::new();
                let work = if let Some(work) = work {
                    assert!(if side == b'L' {
                        work.len() as i32 >= std::cmp::max(1, n)
                    } else {
                        work.len() as i32 >= std::cmp::max(1, m)
                    });
                    work
                } else {
                    let len = <Self as Ormqr>::ormqr_query_work(side, trans, m, n, k) as usize;
                    my_work.resize(len, <Self as Zero>::zero());
                    &mut my_work
                };

                let mut info = 0;
                unsafe {
                    $ormqr(
                        side,
                        trans,
                        m,
                        n,
                        k,
                        a,
                        lda,
                        tau,
                        c,
                        ldc,
                        work,
                        work.len() as i32,
                        &mut info,
                    )
                };
                info
            }

            fn ormqr_query_work(side: u8, trans: u8, m: i32, n: i32, k: i32) -> i32 {
                let a = [<Self as Zero>::zero()];
                let tau = [<Self as Zero>::zero()];
                let mut c = [<Self as Zero>::zero()];
                let mut work_query = [<Self as Zero>::zero()];
                let mut info = 0;

                assert!(side == b'L' || side == b'R');
                assert!(trans == b'T' || trans == b'N');
                assert!(m >= 0);
                assert!(n >= 0);
                assert!(if side == b'L' {
                    k >= 0 && k <= m
                } else {
                    k >= 0 && k <= n
                });
                let lda = if side == b'L' { m } else { n };
                unsafe {
                    $ormqr(
                        side,
                        trans,
                        m,
                        n,
                        k,
                        &a,
                        lda,
                        &tau,
                        &mut c,
                        m,
                        &mut work_query,
                        -1,
                        &mut info,
                    );
                }
                assert_eq!(info, 0);
                work_query[0].re() as i32
            }
        }
    };
}

impl_ormqr_real!(f32, sormqr);
impl_ormqr_real!(f64, dormqr);
impl_ormqr_complex!(c32, cunmqr);
impl_ormqr_complex!(c64, zunmqr);
