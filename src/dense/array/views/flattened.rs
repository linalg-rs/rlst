//! Flattened view onto an array

use crate::dense::layout::convert_1d_nd_from_shape;

use crate::dense::array::Array;

use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, Shape, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::RlstBase;

/// A flattened view onto an array.
///
/// Use the funtion [arr.flattened()](crate::Array::view_flat) instead.
pub struct ArrayFlatView<
    'a,
    Item: RlstBase,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    /// Instantiate a new flatted view.
    pub fn new(arr: &'a Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<1> for ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; 1] {
        [self.arr.shape().iter().product()]
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<1> for ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        let new_index = convert_1d_nd_from_shape(multi_index[0], self.arr.shape());
        self.arr.get_value_unchecked(new_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<1> for ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        let new_index = convert_1d_nd_from_shape(multi_index[0], self.arr.shape());
        self.arr.get_unchecked(new_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayFlatView<'a, Item, ArrayImpl, NDIM>
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
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}

////////// Mutable flattened view

/// A flattened view onto an array.
///
/// Use the funtion [arr.flattened_mut()](crate::Array::view_flat_mut) instead.
pub struct ArrayFlatViewMut<
    'a,
    Item: RlstBase,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    /// Instantiate a new flattened view.
    pub fn new(arr: &'a mut Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Shape<1> for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; 1] {
        [self.arr.shape().iter().product()]
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<1> for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        let new_index = convert_1d_nd_from_shape(multi_index[0], self.arr.shape());
        self.arr.get_value_unchecked(new_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<1> for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; 1]) -> &mut Self::Item {
        let new_index = convert_1d_nd_from_shape(multi_index[0], self.arr.shape());
        self.arr.get_unchecked_mut(new_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<1> for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        let new_index = convert_1d_nd_from_shape(multi_index[0], self.arr.shape());
        self.arr.get_unchecked(new_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
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
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > RawAccessMut for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        self.arr.buff_ptr_mut()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}
