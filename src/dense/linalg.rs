//! Linear algebra routines

pub mod inverse;
pub mod lu;
pub mod pseudo_inverse;
pub mod qr;
pub mod svd;

use crate::dense::linalg::inverse::MatrixInverse;
use crate::dense::linalg::lu::MatrixLu;
use crate::dense::linalg::pseudo_inverse::MatrixPseudoInverse;
use crate::dense::linalg::qr::MatrixQr;
use crate::dense::linalg::svd::MatrixSvd;

/// Return true if stride is column major as required by Lapack.
pub fn assert_lapack_stride(stride: [usize; 2]) {
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );
}

/// Marker trait for objects that support Matrix decompositions.
pub trait LinAlg: MatrixInverse + MatrixQr + MatrixSvd + MatrixLu + MatrixPseudoInverse {}

impl<T: MatrixInverse + MatrixQr + MatrixSvd + MatrixLu + MatrixPseudoInverse> LinAlg for T {}

// // Implementation of LinAlg Decomposition traits for views

// impl<
//         'a,
//         Item: RlstScalar,
//         ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
//             + Shape<2>
//             + UnsafeRandomAccessMut<2, Item = Item>,
//     > MatrixInverse for Array<Item, ArrayViewMut<'a, Item, ArrayImpl, 2>, 2>
// where
//     Array<Item, ArrayImpl, 2>: MatrixInverse,
// {
//     fn into_inverse_alloc(self) -> crate::RlstResult<()> {
//         todo!()
//     }
// }
