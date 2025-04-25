//! Subview onto an array
use crate::dense::array::views::{compute_raw_range, offset_multi_index};
use crate::dense::array::Array;

use crate::dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape};

use crate::dense::traits::{
    RawAccess, RawAccessMut, Shape, Stride, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
    UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};

/// Subview of an array
pub struct ArraySubView<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> ArraySubView<ArrayImpl, NDIM> {
    /// Create new array sub-view
    pub fn new(arr: Array<ArrayImpl, NDIM>, offset: [usize; NDIM], shape: [usize; NDIM]) -> Self {
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

impl<ArrayImpl, const NDIM: usize> Shape<NDIM> for ArraySubView<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArraySubView<ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArraySubView<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) = compute_raw_range(self.offset, self.stride(), self.shape());

        &self.arr.data()[start_raw..end_raw]
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_value_unchecked(offset_multi_index(multi_index, self.offset))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.arr
            .get_unchecked(offset_multi_index(multi_index, self.offset))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for ArraySubView<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        let (start_raw, end_raw) =
            crate::dense::array::views::compute_raw_range(self.offset, self.stride(), self.shape());

        &mut self.arr.data_mut()[start_raw..end_raw]
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
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
