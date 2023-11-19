//! Computational BLIS routines
use super::types::TransMode;
use crate::interface::assert_data_size;
use crate::raw;
use cauchy::{c32, c64, Scalar};

pub trait Gemm: Scalar {
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        transa: TransMode,
        transb: TransMode,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        rsa: usize,
        csa: usize,
        b: &[Self],
        rsb: usize,
        csb: usize,
        beta: Self,
        c: &mut [Self],
        rsc: usize,
        csc: usize,
    );
}

macro_rules! impl_gemm {
    ($scalar:ty, $blis_gemm:ident, $blis_type:ty) => {
        impl Gemm for $scalar {
            fn gemm(
                transa: TransMode,
                transb: TransMode,
                m: usize,
                n: usize,
                k: usize,
                alpha: Self,
                a: &[Self],
                rsa: usize,
                csa: usize,
                b: &[Self],
                rsb: usize,
                csb: usize,
                beta: Self,
                c: &mut [Self],
                rsc: usize,
                csc: usize,
            ) {
                match transa {
                    TransMode::NoTrans => assert_data_size(a, [rsa, csa], [m, k]),
                    TransMode::ConjNoTrans => assert_data_size(a, [rsa, csa], [m, k]),
                    TransMode::Trans => assert_data_size(a, [rsa, csa], [k, m]),
                    TransMode::ConjTrans => assert_data_size(a, [rsa, csa], [k, m]),
                }

                match transb {
                    TransMode::NoTrans => assert_data_size(b, [rsb, csb], [k, n]),
                    TransMode::ConjNoTrans => assert_data_size(b, [rsb, csb], [k, n]),
                    TransMode::Trans => assert_data_size(b, [rsb, csb], [n, k]),
                    TransMode::ConjTrans => assert_data_size(b, [rsb, csb], [n, k]),
                }

                assert_data_size(c, [rsc, csc], [m, n]);

                unsafe {
                    raw::$blis_gemm(
                        transa as u32,
                        transb as u32,
                        m as i64,
                        n as i64,
                        k as i64,
                        &alpha as *const _ as *const $blis_type,
                        a.as_ptr() as *const _ as *const $blis_type,
                        rsa as i64,
                        csa as i64,
                        b.as_ptr() as *const _ as *const $blis_type,
                        rsb as i64,
                        csb as i64,
                        &beta as *const _ as *const $blis_type,
                        c.as_mut_ptr() as *mut _ as *mut $blis_type,
                        rsc as i64,
                        csc as i64,
                    )
                };
            }
        }
    };
}

impl_gemm!(f32, bli_sgemm, f32);
impl_gemm!(f64, bli_dgemm, f64);
impl_gemm!(c32, bli_cgemm, raw::scomplex);
impl_gemm!(c64, bli_zgemm, raw::dcomplex);
