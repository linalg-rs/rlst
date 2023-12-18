//! Linear algebra routines

use crate::array::Array;
use crate::traits::*;
use rlst_common::types::Scalar;

use self::{
    inverse::MatrixInverse, lu::MatrixLuDecomposition, pseudo_inverse::MatrixPseudoInverse,
    qr::MatrixQrDecomposition, svd::MatrixSvd,
};
pub mod inverse;
pub mod lu;
pub mod pseudo_inverse;
pub mod qr;
pub mod svd;

/// Return true if stride is column major as required by Lapack.
pub fn assert_lapack_stride(stride: [usize; 2]) {
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );
}

/// Marker trait for Arrays that provide
pub trait Linalg {}

impl<Item: Scalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>> Linalg
    for Array<Item, ArrayImpl, 2>
where
    Array<Item, ArrayImpl, 2>: MatrixInverse
        + MatrixLuDecomposition
        + MatrixPseudoInverse
        + MatrixQrDecomposition
        + MatrixSvd,
{
}
