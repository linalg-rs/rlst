//! Multiplication of Arrays

use rlst_blis::interface::{gemm::Gemm, types::TransMode};

use super::{empty_axis::AxisPosition, *};

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
    pub fn matmul_into<
        ArrayImplA: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
        ArrayImplB: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    >(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplA, 2>,
        arr_b: Array<Item, ArrayImplB, 2>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;
        crate::matrix_multiply::matrix_multiply(transa, transb, alpha, &arr_a, &arr_b, beta, self)
    }
}

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 1>
{
    pub fn matvec_into<
        ArrayImplA: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
        ArrayImplB: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1> + Stride<1> + RawAccess<Item = Item>,
    >(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplA, 2>,
        arr_b: Array<Item, ArrayImplB, 1>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let mut arr_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Back);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a,
            &arr_b.view().insert_empty_axis(AxisPosition::Back),
            beta,
            &mut arr_with_padded_dim,
        )
    }
}

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 1>
{
    pub fn row_matvec_into<
        ArrayImplA: UnsafeRandomAccessByValue<1, Item = Item> + Shape<1> + Stride<1> + RawAccess<Item = Item>,
        ArrayImplB: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + Stride<2> + RawAccess<Item = Item>,
    >(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplA, 1>,
        arr_b: Array<Item, ArrayImplB, 2>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let mut arr_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Front);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a.view().insert_empty_axis(AxisPosition::Front),
            &arr_b,
            beta,
            &mut arr_with_padded_dim,
        )
    }
}
