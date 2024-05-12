//! Array slicing.

use crate::dense::{
    layout::{convert_1d_nd_from_shape, convert_nd_raw},
    number_types::{IsGreaterByOne, IsGreaterZero, NumberType},
};

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::external::metal::metal_array::{AsRawMetalBuffer, AsRawMetalBufferMut};

use super::{
    empty_chunk, Array, ChunkedAccess, RawAccess, RawAccessMut, RlstScalar, Shape, Stride,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Generic structure to store Array slices.
pub struct ArraySlice<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    arr: Array<Item, ArrayImpl, ADIM>,
    // The first entry is the axis, the second is the index in the axis.
    slice: [usize; 2],
    mask: [usize; NDIM],
}

// Implementation of ArraySlice

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    /// Create new array slice
    pub fn new(arr: Array<Item, ArrayImpl, ADIM>, slice: [usize; 2]) -> Self {
        // The mask is zero for all entries before the sliced out one and
        // one for all entries after.
        let mut mask = [1; NDIM];
        assert!(
            slice[0] < ADIM,
            "Axis {} out of bounds. Array has {} axes.",
            slice[0],
            ADIM
        );
        assert!(
            slice[1] < arr.shape()[slice[0]],
            "Index {} in axis {} out of bounds. Dimension of axis is {}.",
            slice[1],
            slice[0],
            arr.shape()[slice[0]]
        );
        mask.iter_mut().take(slice[0]).for_each(|val| *val = 0);
        Self { arr, slice, mask }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_value_unchecked(orig_index)
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessByRef<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked(orig_index)
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Shape<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_shape = self.arr.shape();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_shape[index + self.mask[index]];
        }

        result
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccess<Item = Item>
            + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccess for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;
    fn data(&self) -> &[Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) =
            compute_raw_range(self.slice, self.arr.stride(), self.arr.shape());

        &self.arr.data()[start_raw..end_raw]
    }

    fn buff_ptr(&self) -> *const Self::Item {
        self.arr.buff_ptr()
    }

    fn offset(&self) -> usize {
        let mut orig_index = [0; ADIM];
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.offset() + convert_nd_raw(orig_index, self.arr.stride())
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + Stride<ADIM>
            + AsRawMetalBuffer,
        const ADIM: usize,
        const NDIM: usize,
    > AsRawMetalBuffer for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn metal_buffer(&self) -> &crate::external::metal::interface::MetalBuffer {
        self.arr.metal_buffer()
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + UnsafeRandomAccessMut<ADIM, Item = Item>
            + Shape<ADIM>
            + Stride<ADIM>
            + AsRawMetalBufferMut,
        const ADIM: usize,
        const NDIM: usize,
    > AsRawMetalBufferMut for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn metal_buffer_mut(&mut self) -> &mut crate::external::metal::interface::MetalBuffer {
        self.arr.metal_buffer_mut()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM> + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Stride<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride: [usize; ADIM] = self.arr.stride();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_stride[index + self.mask[index]];
        }

        result
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
    > Array<Item, ArrayImpl, ADIM>
{
    /// Create a slice from a given array.
    ///
    /// Consider an array `arr` with shape `[a0, a1, a2, a3, ...]`. The function call
    /// `arr.slice(2, 3)` returns a one dimension smaller array indexed by `[a0, a1, 3, a3, ...]`.
    /// Hence, the dimension `2` has been fixed to always have the value `3.`
    ///
    /// # Examples
    ///
    /// If `arr` is a matrix then the first column of the matrix is obtained from
    /// `arr.slice(1, 0)`, while the third row of the matrix is obtained from
    /// `arr.slice(0, 2)`.
    pub fn slice<const NDIM: usize>(
        self,
        axis: usize,
        index: usize,
    ) -> Array<Item, ArraySlice<Item, ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsGreaterByOne<NDIM>,
        NumberType<NDIM>: IsGreaterZero,
    {
        Array::new(ArraySlice::new(self, [axis, index]))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM> + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        let nelements = self.shape().iter().product();
        if let Some(mut chunk) = empty_chunk(chunk_index, nelements) {
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

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessByRef<ADIM, Item = Item>
            + UnsafeRandomAccessMut<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked_mut(orig_index)
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccessMut<Item = Item>
            + Stride<ADIM>
            + UnsafeRandomAccessMut<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccessMut for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) =
            compute_raw_range(self.slice, self.arr.stride(), self.arr.shape());
        &mut self.arr.data_mut()[start_raw..end_raw]
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        self.arr.buff_ptr_mut()
    }
}

// ////////////////////

fn multi_index_to_orig<const ADIM: usize, const NDIM: usize>(
    multi_index: [usize; NDIM],
    mask: [usize; NDIM],
) -> [usize; ADIM]
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    let mut orig = [0; ADIM];
    for (index, &value) in multi_index.iter().enumerate() {
        orig[index + mask[index]] = value;
    }
    orig
}

fn compute_raw_range<const NDIM: usize>(
    slice: [usize; 2],
    stride: [usize; NDIM],
    shape: [usize; NDIM],
) -> (usize, usize) {
    let mut start_multi_index = [0; NDIM];
    start_multi_index[slice[0]] = slice[1];
    let mut end_multi_index = shape;
    for (index, value) in end_multi_index.iter_mut().enumerate() {
        if index == slice[0] {
            *value = slice[1]
        } else {
            // We started with the shape. Reduce
            // each value of the shape by 1 to get last
            // index in that dimension.
            assert!(*value > 0);
            *value -= 1;
        }
    }
    (
        convert_nd_raw(start_multi_index, stride),
        1 + convert_nd_raw(end_multi_index, stride),
    )
}
