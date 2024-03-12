//! Blis implementation of Gemm

use crate::gemm::Gemm;
use crate::types::{c32, c64, TransMode};

/// Compute expected size of a data slice from stride and shape.
fn get_expected_data_size(stride: [usize; 2], shape: [usize; 2]) -> usize {
    if shape[0] == 0 || shape[1] == 0 {
        return 0;
    }

    1 + (shape[0] - 1) * stride[0] + (shape[1] - 1) * stride[1]
}

/// Panic if expected data size is not identical to actual data size.
fn assert_data_size(nelems: usize, stride: [usize; 2], shape: [usize; 2]) {
    let expected = get_expected_data_size(stride, shape);

    assert_eq!(
        expected, nelems,
        "Wrong size for data slice. Actual size {}. Expected size {}.",
        nelems, expected
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
                std::unimplemented!();
                //             match transa {
                //                 TransMode::NoTrans => assert_data_size(a.len(), [rsa, csa], [m, k]),
                //                 TransMode::ConjNoTrans => assert_data_size(a.len(), [rsa, csa], [m, k]),
                //                 TransMode::Trans => assert_data_size(a.len(), [rsa, csa], [k, m]),
                //                 TransMode::ConjTrans => assert_data_size(a.len(), [rsa, csa], [k, m]),
                //             }

                //             match transb {
                //                 TransMode::NoTrans => assert_data_size(b.len(), [rsb, csb], [k, n]),
                //                 TransMode::ConjNoTrans => assert_data_size(b.len(), [rsb, csb], [k, n]),
                //                 TransMode::Trans => assert_data_size(b.len(), [rsb, csb], [n, k]),
                //                 TransMode::ConjTrans => assert_data_size(b.len(), [rsb, csb], [n, k]),
                //             }

                //             assert_data_size(c.len(), [rsc, csc], [m, n]);

                //             unsafe {
                //                 raw::$blis_gemm(
                //                     convert_trans_mode(transa),
                //                     convert_trans_mode(transb),
                //                     m as i64,
                //                     n as i64,
                //                     k as i64,
                //                     &alpha as *const _ as *const $blis_type,
                //                     a.as_ptr() as *const _ as *const $blis_type,
                //                     rsa as i64,
                //                     csa as i64,
                //                     b.as_ptr() as *const _ as *const $blis_type,
                //                     rsb as i64,
                //                     csb as i64,
                //                     &beta as *const _ as *const $blis_type,
                //                     c.as_mut_ptr() as *mut _ as *mut $blis_type,
                //                     rsc as i64,
                //                     csc as i64,
                //                 )
                //             };
            }
        }
    };
}

impl_gemm!(f32, bli_sgemm, f32);
impl_gemm!(f64, bli_dgemm, f64);
impl_gemm!(c32, bli_cgemm, raw::scomplex);
impl_gemm!(c64, bli_zgemm, raw::dcomplex);
