//! Default view onto an array.

use crate::dense::types::RlstBase;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::external::metal::metal_array::AsRawMetalBufferMut;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::AsRawMetalBuffer;

use crate::dense::array::Array;

use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Basic structure for a `View`
pub struct ArrayView<
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
    > ArrayView<'a, Item, ArrayImpl, NDIM>
{
    /// Create new view
    pub fn new(arr: &'a Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + Stride<NDIM>,
        const NDIM: usize,
    > Stride<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<
        'a,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f32> + Shape<NDIM> + Stride<NDIM> + AsRawMetalBuffer,
        const NDIM: usize,
    > AsRawMetalBuffer for ArrayView<'a, f32, ArrayImpl, NDIM>
{
    fn metal_buffer(&self) -> &crate::external::metal::interface::MetalBuffer {
        self.arr.metal_buffer()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + RawAccess<Item = Item>
            + Stride<NDIM>,
        const NDIM: usize,
    > RawAccess for ArrayView<'a, Item, ArrayImpl, NDIM>
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
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(multi_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayView<'a, Item, ArrayImpl, NDIM>
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
    > ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    /// Create new mutable view
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
    > Shape<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + Stride<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Stride<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + RawAccess<Item = Item>
            + Stride<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > RawAccess for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
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
            + RawAccess<Item = Item>
            + Stride<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > RawAccessMut for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
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
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(multi_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + ChunkedAccess<N, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        self.arr.get_chunk(chunk_index)
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr.get_unchecked_mut(multi_index)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<
        'a,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f32>
            + UnsafeRandomAccessMut<NDIM, Item = f32>
            + Shape<NDIM>
            + Stride<NDIM>
            + AsRawMetalBuffer,
        const NDIM: usize,
    > AsRawMetalBuffer for ArrayViewMut<'a, f32, ArrayImpl, NDIM>
{
    fn metal_buffer(&self) -> &crate::external::metal::interface::MetalBuffer {
        self.arr.metal_buffer()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<
        'a,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f32>
            + Shape<NDIM>
            + Stride<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = f32>
            + AsRawMetalBufferMut,
        const NDIM: usize,
    > AsRawMetalBufferMut for ArrayViewMut<'a, f32, ArrayImpl, NDIM>
{
    fn metal_buffer_mut(&mut self) -> &mut crate::external::metal::interface::MetalBuffer {
        self.arr.metal_buffer_mut()
    }
}

impl<
        'a,
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + ResizeInPlace<NDIM>,
        const NDIM: usize,
    > ResizeInPlace<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    #[inline]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.arr.resize_in_place(shape)
    }
}
