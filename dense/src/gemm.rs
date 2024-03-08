//! Gemm trait for matrix multiplication
use crate::types::TransMode;

pub trait Gemm: Sized {
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

pub mod blis;
