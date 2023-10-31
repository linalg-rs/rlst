//! Multiplication of Arrays

use rlst_blis::interface::{gemm::Gemm, types::TransMode};
use rlst_common::traits::MultInto;
use rlst_common::traits::MultIntoResize;

use super::{empty_axis::AxisPosition, *};

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
        ArrayImplFirst: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > MultInto<Array<Item, ArrayImplFirst, 2>, Array<Item, ArrayImplSecond, 2>>
    for Array<Item, ArrayImpl, 2>
{
    type Item = Item;

    fn mult_into(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 2>,
        arr_b: Array<Item, ArrayImplSecond, 2>,
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
        ArrayImplFirst: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
    > MultInto<Array<Item, ArrayImplFirst, 2>, Array<Item, ArrayImplSecond, 1>>
    for Array<Item, ArrayImpl, 1>
{
    type Item = Item;

    fn mult_into(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 2>,
        arr_b: Array<Item, ArrayImplSecond, 1>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let mut self_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Back);
        let arr_with_padded_dim = arr_b.view().insert_empty_axis(AxisPosition::Back);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a,
            &arr_with_padded_dim,
            beta,
            &mut self_with_padded_dim,
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
        ArrayImplFirst: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > MultInto<Array<Item, ArrayImplFirst, 1>, Array<Item, ArrayImplSecond, 2>>
    for Array<Item, ArrayImpl, 1>
{
    type Item = Item;

    fn mult_into(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 1>,
        arr_b: Array<Item, ArrayImplSecond, 2>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let mut self_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Front);
        let arr_with_padded_dim = arr_a.view().insert_empty_axis(AxisPosition::Front);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_with_padded_dim,
            &arr_b,
            beta,
            &mut self_with_padded_dim,
        )
    }
}

/// MultIntoResize

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + ResizeInPlace<2>,
        ArrayImplFirst: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > MultIntoResize<Array<Item, ArrayImplFirst, 2>, Array<Item, ArrayImplSecond, 2>>
    for Array<Item, ArrayImpl, 2>
{
    type Item = Item;

    fn mult_into_resize(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 2>,
        arr_b: Array<Item, ArrayImplSecond, 2>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let expected_shape = [arr_a.shape()[0], arr_b.shape()[1]];
        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        crate::matrix_multiply::matrix_multiply(transa, transb, alpha, &arr_a, &arr_b, beta, self)
    }
}

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>
            + ResizeInPlace<1>,
        ArrayImplFirst: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
    > MultIntoResize<Array<Item, ArrayImplFirst, 2>, Array<Item, ArrayImplSecond, 1>>
    for Array<Item, ArrayImpl, 1>
{
    type Item = Item;

    fn mult_into_resize(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 2>,
        arr_b: Array<Item, ArrayImplSecond, 1>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let expected_shape = [arr_a.shape()[0]];

        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        let mut self_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Back);
        let arr_with_padded_dim = arr_b.view().insert_empty_axis(AxisPosition::Back);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a,
            &arr_with_padded_dim,
            beta,
            &mut self_with_padded_dim,
        )
    }
}

impl<
        Item: Scalar + Gemm,
        ArrayImpl: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>
            + ResizeInPlace<1>,
        ArrayImplFirst: UnsafeRandomAccessByValue<1, Item = Item>
            + UnsafeRandomAccessMut<1, Item = Item>
            + Shape<1>
            + Stride<1>
            + RawAccessMut<Item = Item>,
        ArrayImplSecond: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>
            + Stride<2>
            + RawAccessMut<Item = Item>,
    > MultIntoResize<Array<Item, ArrayImplFirst, 1>, Array<Item, ArrayImplSecond, 2>>
    for Array<Item, ArrayImpl, 1>
{
    type Item = Item;

    fn mult_into_resize(
        &mut self,
        alpha: Item,
        arr_a: Array<Item, ArrayImplFirst, 1>,
        arr_b: Array<Item, ArrayImplSecond, 2>,
        beta: Item,
    ) {
        let transa = TransMode::NoTrans;
        let transb = TransMode::NoTrans;

        let expected_shape = [arr_b.shape()[1]];

        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        let mut self_with_padded_dim = self.view_mut().insert_empty_axis(AxisPosition::Front);
        let arr_with_padded_dim = arr_a.view().insert_empty_axis(AxisPosition::Front);

        crate::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_with_padded_dim,
            &arr_b,
            beta,
            &mut self_with_padded_dim,
        )
    }
}
