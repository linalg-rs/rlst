//! Extend an array by an empty axis either at the front or back.

use crate::dense::{
    number_types::{IsSmallerByOne, NumberType},
    traits::{UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut},
    types::RlstBase,
};

use super::{
    Array, ChunkedAccess, DataChunk, RawAccess, RawAccessMut, Shape, Stride,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Axis position.
#[derive(Clone, Copy)]
pub enum AxisPosition {
    /// Insert axis at the front.
    Front,
    /// Insert axis at the back.
    Back,
}

/// Array implementation that provides an appended empty axis.
pub struct ArrayAppendAxis<
    Item: RlstBase,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    arr: Array<Item, ArrayImpl, ADIM>,
    axis_position: AxisPosition,
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    /// Create new
    pub fn new(arr: Array<Item, ArrayImpl, ADIM>, axis_position: AxisPosition) -> Self {
        Self { arr, axis_position }
    }
}

impl<
        Item: RlstBase,
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
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandom1DAccessByValue<Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandom1DAccessByRef<Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessByRef for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandom1DAccessMut<Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessMut for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(index)
    }
}

impl<
        Item: RlstBase,
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

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr
            .get_unchecked(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<
        Item: RlstBase,
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

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr
            .get_unchecked_mut(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<
        Item: RlstBase,
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
        Item: RlstBase,
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

    fn buff_ptr(&self) -> *const Self::Item {
        self.arr.buff_ptr()
    }

    fn offset(&self) -> usize {
        self.arr.offset()
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccessMut<Item = Item>
            + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccessMut for ArrayAppendAxis<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        self.arr.buff_ptr_mut()
    }
}

impl<
        Item: RlstBase,
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
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
    > Array<Item, ArrayImpl, ADIM>
{
    /// Insert empty axis
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
        Item: RlstBase,
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

/// Map an extended index of dimension NDIM to an index of dimension ADIM with NDIM = ADIM + 1.
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
