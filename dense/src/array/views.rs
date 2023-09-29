//! Views onto an array

use crate::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape};

use super::Array;
use rlst_common::traits::*;

pub struct ArrayView<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

pub struct ArrayViewMut<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayView<'a, Item, ArrayImpl, NDIM>
{
    pub fn new(
        arr: &'a Array<Item, ArrayImpl, NDIM>,
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

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    pub fn new(
        arr: &'a mut Array<Item, ArrayImpl, NDIM>,
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

// Basic traits for ArrayView

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
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
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
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
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<rlst_common::types::DataChunk<Self::Item, N>> {
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
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
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
        'a,
        Item: Scalar,
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
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_unchecked(offset_multi_index(multi_index, self.offset))
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<rlst_common::types::DataChunk<Self::Item, N>> {
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

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
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
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn subview(
        &self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayView::new(self, offset, shape))
    }

    pub fn view(&self) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        self.subview([0; NDIM], self.shape())
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
    pub fn subview_mut(
        &mut self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayViewMut::new(self, offset, shape))
    }

    pub fn view_mut(&mut self) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        self.subview_mut([0; NDIM], self.shape())
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
