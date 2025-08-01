//! Implementation of array multiplication.

use crate::{
    base_types::TransMode,
    traits::linalg::base::{Gemm, MultInto, MultIntoResize},
    AsOwnedRefType, AsOwnedRefTypeMut,
};

use super::{
    empty_axis::AxisPosition, Array, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride,
};

impl<
        Item: Gemm,
        ArrayImpl: RawAccessMut<Item = Item> + Stride<2> + Shape<2>,
        ArrayImplFirst: RawAccess<Item = Item> + Stride<2> + Shape<2>,
        ArrayImplSecond: RawAccess<Item = Item> + Stride<2> + Shape<2>,
    > MultInto<Array<ArrayImplFirst, 2>, Array<ArrayImplSecond, 2>> for Array<ArrayImpl, 2>
{
    fn mult_into(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 2>,
        arr_b: Array<ArrayImplSecond, 2>,
        beta: Item,
    ) -> Self {
        crate::dense::matrix_multiply::matrix_multiply(
            transa, transb, alpha, &arr_a, &arr_b, beta, &mut self,
        );
        self
    }
}

impl<
        Item: Gemm,
        ArrayImpl: RawAccessMut<Item = Item> + Stride<1> + Shape<1>,
        ArrayImplFirst: RawAccess<Item = Item> + Stride<2> + Shape<2>,
        ArrayImplSecond: RawAccess<Item = Item> + Stride<1> + Shape<1>,
    > MultInto<Array<ArrayImplFirst, 2>, Array<ArrayImplSecond, 1>> for Array<ArrayImpl, 1>
{
    fn mult_into(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 2>,
        arr_b: Array<ArrayImplSecond, 1>,
        beta: Item,
    ) -> Self {
        let mut self_with_padded_dim = self.r_mut().insert_empty_axis(AxisPosition::Back);
        let arr_with_padded_dim = arr_b.r().insert_empty_axis(AxisPosition::Back);

        crate::dense::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a,
            &arr_with_padded_dim,
            beta,
            &mut self_with_padded_dim,
        );
        self
    }
}

impl<
        Item: Gemm,
        ArrayImpl: RawAccessMut<Item = Item> + Stride<1> + Shape<1>,
        ArrayImplFirst: Shape<1> + Stride<1> + RawAccess<Item = Item>,
        ArrayImplSecond: Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > MultInto<Array<ArrayImplFirst, 1>, Array<ArrayImplSecond, 2>> for Array<ArrayImpl, 1>
{
    fn mult_into(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 1>,
        arr_b: Array<ArrayImplSecond, 2>,
        beta: Item,
    ) -> Self {
        let mut self_with_padded_dim = self.r_mut().insert_empty_axis(AxisPosition::Front);
        let arr_with_padded_dim = arr_a.r().insert_empty_axis(AxisPosition::Front);

        crate::dense::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_with_padded_dim,
            &arr_b,
            beta,
            &mut self_with_padded_dim,
        );
        self
    }
}

// MultIntoResize

impl<
        Item: Gemm,
        ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Item> + ResizeInPlace<2>,
        ArrayImplFirst: Shape<2> + Stride<2> + RawAccess<Item = Item>,
        ArrayImplSecond: Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > MultIntoResize<Array<ArrayImplFirst, 2>, Array<ArrayImplSecond, 2>> for Array<ArrayImpl, 2>
{
    fn mult_into_resize(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 2>,
        arr_b: Array<ArrayImplSecond, 2>,
        beta: Item,
    ) -> Self {
        let shapea = new_shape(arr_a.shape(), transa);
        let shapeb = new_shape(arr_b.shape(), transb);

        let expected_shape = [shapea[0], shapeb[1]];

        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        crate::dense::matrix_multiply::matrix_multiply(
            transa, transb, alpha, &arr_a, &arr_b, beta, &mut self,
        );
        self
    }
}

impl<
        Item: Gemm,
        ArrayImpl: Shape<1> + Stride<1> + RawAccessMut<Item = Item> + ResizeInPlace<1>,
        ArrayImplFirst: Shape<2> + Stride<2> + RawAccess<Item = Item>,
        ArrayImplSecond: Shape<1> + Stride<1> + RawAccess<Item = Item>,
    > MultIntoResize<Array<ArrayImplFirst, 2>, Array<ArrayImplSecond, 1>> for Array<ArrayImpl, 1>
{
    fn mult_into_resize(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 2>,
        arr_b: Array<ArrayImplSecond, 1>,
        beta: Item,
    ) -> Self {
        let shapea = new_shape(arr_a.shape(), transa);

        let expected_shape = [shapea[0]];

        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        let mut self_with_padded_dim = self.r_mut().insert_empty_axis(AxisPosition::Back);
        let arr_with_padded_dim = arr_b.r().insert_empty_axis(AxisPosition::Back);

        crate::dense::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_a,
            &arr_with_padded_dim,
            beta,
            &mut self_with_padded_dim,
        );
        self
    }
}

impl<
        Item: Gemm,
        ArrayImpl: Shape<1> + Stride<1> + RawAccessMut<Item = Item> + ResizeInPlace<1>,
        ArrayImplFirst: Shape<1> + Stride<1> + RawAccess<Item = Item>,
        ArrayImplSecond: Shape<2> + Stride<2> + RawAccess<Item = Item>,
    > MultIntoResize<Array<ArrayImplFirst, 1>, Array<ArrayImplSecond, 2>> for Array<ArrayImpl, 1>
{
    fn mult_into_resize(
        mut self,
        transa: TransMode,
        transb: TransMode,
        alpha: Item,
        arr_a: Array<ArrayImplFirst, 1>,
        arr_b: Array<ArrayImplSecond, 2>,
        beta: Item,
    ) -> Self {
        let shapeb = new_shape(arr_b.shape(), transb);

        let expected_shape = [shapeb[1]];

        if self.shape() != expected_shape {
            self.resize_in_place(expected_shape);
        }

        let mut self_with_padded_dim = self.r_mut().insert_empty_axis(AxisPosition::Front);
        let arr_with_padded_dim = arr_a.r().insert_empty_axis(AxisPosition::Front);

        crate::dense::matrix_multiply::matrix_multiply(
            transa,
            transb,
            alpha,
            &arr_with_padded_dim,
            &arr_b,
            beta,
            &mut self_with_padded_dim,
        );
        self
    }
}

fn new_shape(shape: [usize; 2], trans_mode: TransMode) -> [usize; 2] {
    match trans_mode {
        TransMode::NoTrans => shape,
        TransMode::ConjNoTrans => shape,
        TransMode::Trans => [shape[1], shape[0]],
        TransMode::ConjTrans => [shape[1], shape[0]],
    }
}
