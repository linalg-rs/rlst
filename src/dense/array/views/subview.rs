//! Subview onto an array
use crate::dense::array::views::{compute_raw_range, offset_multi_index};
use crate::dense::array::Array;

use crate::dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw};

use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, Shape, Stride, UnsafeRandom1DAccessByRef,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::RlstBase;

/// Subview of an array
pub struct ArraySubView<
    Item: RlstBase,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        Item: RlstBase,
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
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + Stride<NDIM>,
        const NDIM: usize,
    > Stride<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<
        Item: RlstBase,
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

    fn offset(&self) -> usize {
        let raw_offset = convert_nd_raw(self.offset, self.stride());
        self.arr.offset() + raw_offset
    }

    fn buff_ptr(&self) -> *const Self::Item {
        self.arr.buff_ptr()
    }
}

impl<
        Item: RlstBase,
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
        Item: RlstBase,
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
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandom1DAccessByValue<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandom1DAccessByRef<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByRef for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandom1DAccessMut<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessMut for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(index)
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + ChunkedAccess<N, Item = Item>
            + UnsafeRandom1DAccessByValue<Item = Item>,
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
            if let Some(mut chunk) = crate::dense::array::empty_chunk(chunk_index, nelements) {
                for count in 0..chunk.valid_entries {
                    unsafe {
                        chunk.data[count] = self.get_value_1d_unchecked(chunk.start_index + count)
                    }
                }
                Some(chunk)
            } else {
                None
            }
        }
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + RawAccessMut<Item = Item>
            + crate::Stride<NDIM>,
        const NDIM: usize,
    > RawAccessMut for ArraySubView<Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) =
            crate::dense::array::views::compute_raw_range(self.offset, self.stride(), self.shape());

        &mut self.arr.data_mut()[start_raw..end_raw]
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        self.arr.buff_ptr_mut()
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArraySubView<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(crate::dense::layout::check_multi_index_in_bounds(
            multi_index,
            self.shape()
        ));
        self.arr
            .get_unchecked_mut(crate::dense::array::views::offset_multi_index(
                multi_index,
                self.offset,
            ))
    }
}
