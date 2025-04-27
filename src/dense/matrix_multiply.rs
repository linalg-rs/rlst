//! Implementation of matrix multiplication.
//!
//! This module implements the matrix multiplication. The current implementation
//! uses the [rlst-blis] crate::dense.

use crate::dense::traits::Gemm;
use crate::dense::traits::{RawAccess, RawAccessMut, Shape, Stride};
use crate::dense::types::TransMode;

/// Matrix mulitplication
///
/// TODO: document what this computes
pub fn matrix_multiply<
    Item: Gemm,
    MatA: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatB: RawAccess<Item = Item> + Shape<2> + Stride<2>,
    MatC: RawAccessMut<Item = Item> + Shape<2> + Stride<2>,
>(
    transa: TransMode,
    transb: TransMode,
    alpha: Item,
    mat_a: &MatA,
    mat_b: &MatB,
    beta: Item,
    mat_c: &mut MatC,
) {
    let m = mat_c.shape()[0];
    let n = mat_c.shape()[1];

    let a_shape = match transa {
        TransMode::NoTrans => mat_a.shape(),
        TransMode::ConjNoTrans => mat_a.shape(),
        TransMode::Trans => [mat_a.shape()[1], mat_a.shape()[0]],
        TransMode::ConjTrans => [mat_a.shape()[1], mat_a.shape()[0]],
    };

    let b_shape = match transb {
        TransMode::NoTrans => mat_b.shape(),
        TransMode::ConjNoTrans => mat_b.shape(),
        TransMode::Trans => [mat_b.shape()[1], mat_b.shape()[0]],
        TransMode::ConjTrans => [mat_b.shape()[1], mat_b.shape()[0]],
    };

    assert_eq!(m, a_shape[0], "Wrong dimension. {} != {}", m, a_shape[0]);
    assert_eq!(n, b_shape[1], "Wrong dimension. {} != {}", n, b_shape[1]);
    assert_eq!(
        a_shape[1], b_shape[0],
        "Wrong dimension. {} != {}",
        a_shape[1], b_shape[0]
    );

    let [rsa, csa] = mat_a.stride();
    let [rsb, csb] = mat_b.stride();
    let [rsc, csc] = mat_c.stride();

    <Item as Gemm>::gemm(
        transa,
        transb,
        m,
        n,
        a_shape[1],
        alpha,
        mat_a.data(),
        rsa,
        csa,
        mat_b.data(),
        rsb,
        csb,
        beta,
        mat_c.data_mut(),
        rsc,
        csc,
    )
}
