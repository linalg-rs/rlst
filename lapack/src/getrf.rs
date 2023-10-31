//! Getrf
use lapack::{cgetrf, dgetrf, sgetrf, zgetrf};
use rlst_common::types::*;

pub trait Getrf: Scalar {
    fn getrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32]) -> i32;
}

macro_rules! impl_getrf {
    ($scalar:ty, $getrf:expr) => {
        impl Getrf for $scalar {
            fn getrf(m: i32, n: i32, a: &mut [Self], lda: i32, ipiv: &mut [i32]) -> i32 {
                assert!(m >= 0);
                assert!(m >= 0);
                assert!(lda >= std::cmp::max(m, 1));
                assert_eq!(a.len() as i32, lda * n);
                assert_eq!(ipiv.len() as i32, std::cmp::min(m, n));

                let mut info = 0;

                unsafe { $getrf(m, n, a, lda, ipiv, &mut info) }

                info
            }
        }
    };
}

impl_getrf!(f32, sgetrf);
impl_getrf!(f64, dgetrf);
impl_getrf!(c32, cgetrf);
impl_getrf!(c64, zgetrf);
