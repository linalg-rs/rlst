//! Multiplication of Arrays

use rlst_blis::interface::gemm::Gemm;

use super::*;

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + Shape<NDIM>
            + Stride<NDIM>
            + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    // pub fn multiply_into<
    //     ArrayImplA: UnsafeRandomAccessByValue<NDIMA, Item = Item>
    //         + Shape<NDIMA>
    //         + Stride<NDIMA>
    //         + RawAccess<Item = Item>,
    //     ArrayImplB: UnsafeRandomAccessByValue<NDIMB, Item = Item>
    //         + Shape<NDIMB>
    //         + Stride<NDIMB>
    //         + RawAccess<Item = Item>,
    //     const NDIMA: usize,
    //     const NDIMB: usize,
    // >(
    //     &mut self,
    //     alpha: Item,
    //     mat_a: &Array<Item, ArrayImplA, NDIMA>,
    //     mat_b: &Array<Item, ArrayImplB, NDIMB>,
    //     beta: Item,
    // ) {
    //     if NDIMA == 1 && NDIMB == 1 {
    //         // Inner Product
    //         assert_eq!(NDIM, 1);
    //         assert_eq!(self.shape()[0], 1);

    //         let multi_index = [0; NDIM];
    //         self[multi_index] *= beta;

    //         self[multi_index] = alpha * mat_a.iter().zip(mat_b.iter()).map(|(a, b)| a * b).sum();
    //     } else if NDIMA == 2 && NDIMB == 1 {
    //         // Matvec
    //         crate::matrix_multiply::matrix_multiply(
    //             TransMode::NoTrans,
    //             TransMode::NoTrans,
    //             alpha,
    //             mat_a,
    //             mat_b,
    //             beta,
    //             self,
    //         );
    //     } else if NDIMB == 1 {
    //         // Corresponding Numpy tensor product
    //         std::unimplemented!()
    //     } else if NDIMB > 1 {
    //         // Corresponding Numpy tensor product
    //         std::unimplemented!()
    //     }
    // }
}
