//! Flattened view onto an array

use crate::dense::array::Array;

use crate::dense::traits::{
    ChunkedAccess, MutableArrayImpl, RawAccess, RawAccessMut, RefArrayImpl, Shape,
    UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut, ValueArrayImpl,
};
use crate::dense::types::RlstBase;

/// A flattened view onto an array.
///
/// Use the funtion [arr.flattened()](crate::Array::view_flat) instead.
pub struct ArrayFlatView<
    'a,
    Item: RlstBase,
    ArrayImpl: ValueArrayImpl<NDIM, Item>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
}

impl<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    ArrayFlatView<'a, Item, ArrayImpl, NDIM>
{
    /// Instantiate a new flatted view.
    pub fn new(arr: &'a Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize> Shape<1>
    for ArrayFlatView<'_, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; 1] {
        [self.arr.shape().iter().product()]
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<1> for ArrayFlatView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        self.arr.get_value_1d_unchecked(multi_index[0])
    }
}

impl<Item: RlstBase, ArrayImpl: RefArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<1> for ArrayFlatView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        self.arr.get_1d_unchecked(multi_index[0])
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: ValueArrayImpl<NDIM, Item> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayFlatView<'_, Item, ArrayImpl, NDIM>
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
        ArrayImpl: ValueArrayImpl<NDIM, Item> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayFlatView<'_, Item, ArrayImpl, NDIM>
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
    ArrayImpl: MutableArrayImpl<NDIM, Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
}

impl<'a, Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    ArrayFlatViewMut<'a, Item, ArrayImpl, NDIM>
{
    /// Instantiate a new flattened view.
    pub fn new(arr: &'a mut Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize> Shape<1>
    for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; 1] {
        [self.arr.shape().iter().product()]
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<1> for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        self.arr.get_value_1d_unchecked(multi_index[0])
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessMut<1> for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; 1]) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(multi_index[0])
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RefArrayImpl<NDIM, Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<1> for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        self.arr.get_1d_unchecked(multi_index[0])
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
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
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > RawAccessMut for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
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
        ArrayImpl: MutableArrayImpl<NDIM, Item> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    fn get_chunk(&self, chunk_index: usize) -> Option<crate::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}
