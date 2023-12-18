//! Linear algebra routines

use self::{inverse::MatrixInverse, lu::MatrixLuDecomposition};
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

pub trait Linalg {}

impl<T> Linalg for T where T: MatrixInverse + MatrixLuDecomposition<Item = Self> {}
