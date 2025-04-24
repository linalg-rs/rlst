//! Default view onto an array.

use crate::dense::types::RlstBase;

use crate::dense::array::Array;

use crate::dense::traits::{
    ChunkedAccess, MutableArrayImpl, RawAccess, RawAccessMut, RefArrayImpl, ResizeInPlace, Shape,
    Stride, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut, ValueArrayImpl,
};

/// Basic structure for a `View`
pub struct ArrayView<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
}

impl<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    ArrayView<'a, Item, ArrayImpl, NDIM>
{
    /// Create new view
    pub fn new(arr: &'a Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize> Shape<NDIM>
    for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item> + Stride<NDIM>, const NDIM: usize>
    Stride<NDIM> for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: ValueArrayImpl<NDIM, Item> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayView<'_, Item, ArrayImpl, NDIM>
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

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index)
    }
}

impl<Item: RlstBase, ArrayImpl: RefArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(multi_index)
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandom1DAccessByValue for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<Item: RlstBase, ArrayImpl: RefArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandom1DAccessByRef for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: ValueArrayImpl<NDIM, Item> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayView<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}

/////////// ArrayViewMut

/// Mutable array view
pub struct ArrayViewMut<
    'a,
    Item: RlstBase,
    ArrayImpl: MutableArrayImpl<NDIM, Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
}

impl<'a, Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    /// Create new mutable view
    pub fn new(arr: &'a mut Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize> Shape<NDIM>
    for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandom1DAccessByValue for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RefArrayImpl<NDIM, Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByRef for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandom1DAccessMut for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(index)
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item> + Stride<NDIM>, const NDIM: usize>
    Stride<NDIM> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
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
    > RawAccessMut for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        self.arr.buff_ptr_mut()
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + RefArrayImpl<NDIM, Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(multi_index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr.get_unchecked_mut(multi_index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: MutableArrayImpl<NDIM, Item> + ResizeInPlace<NDIM>,
        const NDIM: usize,
    > ResizeInPlace<NDIM> for ArrayViewMut<'_, Item, ArrayImpl, NDIM>
{
    #[inline]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.arr.resize_in_place(shape)
    }
}
