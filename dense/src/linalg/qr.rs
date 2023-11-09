//! Interface to QR Decomposition

// use super::assert_lapack_stride;
// use crate::array::Array;
// use num::One;
// use rlst_common::traits::*;
// use rlst_common::types::*;
// use rlst_lapack::{Dgeqp3, Ormqr};

// pub struct QRDecomposition<
//     Item: Scalar,
//     ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
// > {
//     arr: Array<Item, ArrayImpl, 2>,
//     tau: Vec<Item>,
//     jpvt: Vec<usize>,
// }

// impl<
//         Item: Scalar + Dgeqp3 + Ormqr,
//         ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
//             + Stride<2>
//             + Shape<2>
//             + RawAccessMut<Item = Item>,
//     > QRDecomposition<Item, ArrayImpl>
// {
//     pub fn new(arr: Array<Item, ArrayImpl, 2>, work: Option<)
// }
