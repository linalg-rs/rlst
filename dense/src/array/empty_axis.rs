//! Explicit casts to arrays with specific number of dimensions.

use crate::number_types::{IsSmallerByOne, NumberType};

use super::*;

#[derive(Clone, Copy)]
pub enum AxisPosition {
    Front,
    Back,
}

pub struct ArrayAppendAxis<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    arr: Array<Item, ArrayImpl, ADIM>,
    // The first entry is the axis, the second is the index in the axis.
    axis_position: AxisPosition,
}

// Implementation of ArraySlice

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    pub fn new(arr: Array<Item, ArrayImpl, ADIM>, axis_position: AxisPosition) -> Self {
        Self { arr, axis_position }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr
            .get_value_unchecked(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessByRef<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr
            .get_unchecked(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessMut<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr
            .get_unchecked_mut(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Shape<NDIM> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [1; NDIM];
        let orig_shape = self.arr.shape();

        match self.axis_position {
            AxisPosition::Front => result[1..].copy_from_slice(&orig_shape),
            AxisPosition::Back => result[..ADIM].copy_from_slice(&orig_shape),
        }
        result
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccess<Item = Item>
            + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccess for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;
    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM> + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Stride<NDIM> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride = self.arr.stride();

        match self.axis_position {
            AxisPosition::Front => {
                result[0] = 1;
                result[1..].copy_from_slice(&orig_stride)
            }
            AxisPosition::Back => {
                result[NDIM - 1] = orig_stride[ADIM - 1] * self.arr.shape()[ADIM - 1];
                result[..ADIM].copy_from_slice(&orig_stride)
            }
        }
        result
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
    > Array<Item, ArrayImpl, ADIM>
{
    pub fn insert_empty_axis<const NDIM: usize>(
        self,
        axis_position: AxisPosition,
    ) -> Array<Item, ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsSmallerByOne<NDIM>,
    {
        Array::new(ArrayAppendAxis::new(self, axis_position))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + Stride<ADIM>
            + ChunkedAccess<N, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}

fn multi_index_to_orig<const ADIM: usize, const NDIM: usize>(
    multi_index: [usize; NDIM],
    axis_position: AxisPosition,
) -> [usize; ADIM]
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    let mut orig = [0; ADIM];
    match axis_position {
        AxisPosition::Front => orig.copy_from_slice(&multi_index[1..]),
        AxisPosition::Back => orig.copy_from_slice(&multi_index[..ADIM]),
    }

    orig
}

#[cfg(test)]
mod test {
    use rlst_common::traits::{Shape, Stride};

    use crate::{array::empty_axis::AxisPosition, rlst_dynamic_array3};

    #[test]
    fn test_insert_axis_back() {
        let shape = [3, 7, 6];
        let mut arr = rlst_dynamic_array3!(f64, shape);
        arr.fill_from_seed_equally_distributed(0);

        let new_arr = arr.view().insert_empty_axis(AxisPosition::Back);

        assert_eq!(new_arr.shape(), [3, 7, 6, 1]);

        assert_eq!(new_arr[[1, 2, 5, 0]], arr[[1, 2, 5]]);

        assert_eq!(new_arr.stride(), [1, 3, 21, 126])
    }

    #[test]
    fn test_insert_axis_front() {
        let shape = [3, 7, 6];
        let mut arr = rlst_dynamic_array3!(f64, shape);
        arr.fill_from_seed_equally_distributed(0);

        let new_arr = arr.view().insert_empty_axis(AxisPosition::Front);

        assert_eq!(new_arr.shape(), [1, 3, 7, 6]);

        assert_eq!(new_arr[[0, 1, 2, 5]], arr[[1, 2, 5]]);

        assert_eq!(new_arr.stride(), [1, 1, 3, 21])
    }
}
