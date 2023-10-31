//! Getrf
use crate::Trans;
use lapack::{cgetrs, dgetrs, sgetrs, zgetrs};
use rlst_common::types::*;

pub trait Getrs: Scalar {
    fn getrs(
        trans: Trans,
        n: i32,
        nrhs: i32,
        a: &[Self],
        lda: i32,
        ipiv: &[i32],
        b: &mut [Self],
        ldb: i32,
    ) -> i32;
}

macro_rules! impl_getrs {
    ($scalar:ty, $getrs:expr) => {
        impl Getrs for $scalar {
            fn getrs(
                trans: Trans,
                n: i32,
                nrhs: i32,
                a: &[Self],
                lda: i32,
                ipiv: &[i32],
                b: &mut [Self],
                ldb: i32,
            ) -> i32 {
                assert!(n >= 0);
                assert!(nrhs >= 0);
                assert!(lda >= std::cmp::max(1, n));
                assert_eq!(a.len() as i32, lda * n);
                assert_eq!(ipiv.len() as i32, n);
                assert_eq!(b.len() as i32, ldb * nrhs);
                assert!(ldb >= std::cmp::max(1, n));

                for &elem in ipiv {
                    assert!(elem >= 1);
                    assert!(elem <= n);
                }

                let mut info = 0;

                unsafe { $getrs(trans as u8, n, nrhs, a, lda, ipiv, b, ldb, &mut info) }

                info
            }
        }
    };
}

impl_getrs!(f64, dgetrs);
impl_getrs!(f32, sgetrs);
impl_getrs!(c64, zgetrs);
impl_getrs!(c32, cgetrs);
