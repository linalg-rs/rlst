//! Operations on arrays
use crate::layout::{convert_1d_nd, stride_from_shape};

use super::*;
use rlst_common::traits::*;

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + DefaultIterator<Item = Item>,
        const NDIM: usize,
    > FillFrom<Other> for Array<Item, ArrayImpl, NDIM>
{
    fn fill_from(&mut self, other: Other) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item = other_item;
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn fill_from_chunked<
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Other,
    ) {
        assert_eq!(self.shape(), other.shape());
        let stride = stride_from_shape(self.shape());

        let mut chunk_index = 0;

        while let Some(chunk) = other.get_chunk(chunk_index) {
            let data_start = N * chunk_index;
            let data_end = N * chunk_index + chunk.valid_entries;

            for (data_index, arr_index) in (data_start..data_end).enumerate() {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd(arr_index, stride)) =
                        chunk.data[data_index];
                }
            }

            chunk_index += 1;
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + DefaultIterator<Item = Item>,
        const NDIM: usize,
    > SumInto<Other> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn sum_into(&mut self, alpha: Self::Item, other: Other) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += alpha * other_item;
        }
    }
}
