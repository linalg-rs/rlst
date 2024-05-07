//! Views onto an array.
//!
//! A view onto an array stores a reference to the array and forwards all method calls to the
//! original array. A subview is similar but restricts to a subpart of the original array.

use crate::dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::external::metal::metal_array::AsRawMetalBufferMut;
use crate::AsRawMetalBuffer;

use super::Array;
use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::RlstScalar;

/// Basic structure for a `View`
pub struct ArrayView<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
}

/// Mutable array view
pub struct ArrayViewMut<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
}

impl<
        'a,
        Item: RlstScalar,
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
        Item: RlstScalar,
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

/////////////////
/// Basic traits for ArrayView

impl<
        'a,
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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
}

impl<
        'a,
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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

////////////////
/// Basic traits for ArrayViewMut

impl<
        'a,
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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
}

impl<
        'a,
        Item: RlstScalar,
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
}

impl<
        'a,
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
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
        Item: RlstScalar,
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

/// Subview of an array
pub struct ArraySubView<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArraySubView<Item, ArrayImpl, NDIM>
{
    /// Create new array sub-view
    pub fn new(
        arr: Array<Item, ArrayImpl, NDIM>,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Self {
        let arr_shape = arr.shape();
        for index in 0..NDIM {
            assert!(
                offset[index] + shape[index] <= arr_shape[index],
                "View out of bounds for dimension {}. {} > {}",
                index,
                offset[index] + shape[index],
                arr_shape[index]
            )
        }
        Self { arr, offset, shape }
    }
}

// Basic traits for ArraySubView

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + Stride<NDIM>,
        const NDIM: usize,
    > Stride<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + RawAccess<Item = Item>
            + Stride<NDIM>,
        const NDIM: usize,
    > RawAccess for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    fn data(&self) -> &[Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) = compute_raw_range(self.offset, self.stride(), self.shape());

        &self.arr.data()[start_raw..end_raw]
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_value_unchecked(offset_multi_index(multi_index, self.offset))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_unchecked(offset_multi_index(multi_index, self.offset))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        if self.offset == [0; NDIM] && self.shape() == self.arr.shape() {
            // If the view is on the full array we can just pass on the chunk request
            self.arr.get_chunk(chunk_index)
        } else {
            // If the view is on a subsection of the array have to recalcuate the chunk
            let nelements = self.shape().iter().product();
            if let Some(mut chunk) = super::empty_chunk(chunk_index, nelements) {
                for count in 0..chunk.valid_entries {
                    unsafe {
                        chunk.data[count] = self.get_value_unchecked(convert_1d_nd_from_shape(
                            chunk.start_index + count,
                            self.shape(),
                        ))
                    }
                }
                Some(chunk)
            } else {
                None
            }
        }
    }
}

// Basic traits for ArrayViewMut

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + RawAccessMut<Item = Item>
            + Stride<NDIM>,
        const NDIM: usize,
    > RawAccessMut for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) = compute_raw_range(self.offset, self.stride(), self.shape());

        &mut self.arr.data_mut()[start_raw..end_raw]
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_unchecked_mut(offset_multi_index(multi_index, self.offset))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Move the array into a subview specified by an offset and shape of the subview.
    pub fn into_subview(
        self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArraySubView<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArraySubView::new(self, offset, shape))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return a view onto the array.
    pub fn view(&self) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayView::new(self))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return a mutable view onto the array.
    pub fn view_mut(&mut self) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayViewMut::new(self))
    }
}

fn offset_multi_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    offset: [usize; NDIM],
) -> [usize; NDIM] {
    let mut output = [0; NDIM];
    for (ind, elem) in output.iter_mut().enumerate() {
        *elem = multi_index[ind] + offset[ind]
    }
    output
}

fn compute_raw_range<const NDIM: usize>(
    offset: [usize; NDIM],
    stride: [usize; NDIM],
    shape: [usize; NDIM],
) -> (usize, usize) {
    use crate::dense::layout::convert_nd_raw;
    let start_multi_index = offset;
    let mut end_multi_index = [0; NDIM];
    for (index, value) in end_multi_index.iter_mut().enumerate() {
        let sum = start_multi_index[index] + shape[index];
        assert!(sum > 0);
        *value = sum - 1;
    }

    let start_raw = convert_nd_raw(start_multi_index, stride);
    let end_raw = convert_nd_raw(end_multi_index, stride);
    (start_raw, 1 + end_raw)
}
