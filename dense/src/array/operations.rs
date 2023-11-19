//! Operations on arrays
use crate::layout::convert_1d_nd_from_shape;

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

        let mut chunk_index = 0;

        while let Some(chunk) = other.get_chunk(chunk_index) {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_start + data_index,
                        self.shape(),
                    )) = chunk.data[data_index];
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
    fn sum_into(&mut self, other: Other) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += other_item;
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
    pub fn sum_into_chunked<
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Other,
    ) where
        Self: ChunkedAccess<N, Item = Item>,
    {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let (Some(mut my_chunk), Some(chunk)) =
            (self.get_chunk(chunk_index), other.get_chunk(chunk_index))
        {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                my_chunk.data[data_index] += chunk.data[data_index];
            }

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_index + data_start,
                        self.shape(),
                    )) = my_chunk.data[data_index];
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
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn set_zero(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::Zero>::zero();
        }
    }

    pub fn set_one(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::One>::one();
        }
    }

    pub fn set_identity(&mut self) {
        self.set_zero();

        for index in 0..self.shape().iter().copied().min().unwrap() {
            *self.get_mut([index; NDIM]).unwrap() = <Item as num::One>::one();
        }
    }
}
