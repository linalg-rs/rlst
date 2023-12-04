//! Implementation of ormqr

use lapack::cunmqr;
use lapack::dormqr;
use lapack::sormqr;
use lapack::zunmqr;
use num::Zero;
use rlst_common::types::*;

use crate::Ormqr;

pub trait Unmqr: Scalar {
    fn unmqr(
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

    fn unmqr_query_work(side: u8, trans: u8, m: i32, n: i32, k: i32) -> i32;
}

impl<T: Ormqr> Unmqr for T {
    fn unmqr(
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
        <Self as Ormqr>::ormqr(side, trans, m, n, k, a, lda, tau, c, ldc, work)
    }

    fn unmqr_query_work(side: u8, trans: u8, m: i32, n: i32, k: i32) -> i32 {
        <Self as Ormqr>::ormqr_query_work(side, trans, m, n, k)
    }
}
