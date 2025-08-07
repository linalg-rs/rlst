//! Subview onto an array
use crate::dense::array::Array;

use crate::dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape};
use crate::traits::{
    accessors::{
        UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
        UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
    },
    base_operations::{BaseItem, Shape, Stride},
};
use crate::{ContainerType, Unknown};

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

impl<ArrayImpl, const NDIM: usize> ContainerType for ArraySubView<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = Unknown;
}

impl<ArrayImpl, const NDIM: usize> BaseItem for ArraySubView<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

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

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
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
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArraySubView<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(crate::dense::layout::check_multi_index_in_bounds(
            multi_index,
            self.shape()
        ));
        self.arr
            .get_unchecked_mut(offset_multi_index(multi_index, self.offset))
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

// fn compute_raw_range<const NDIM: usize>(
//     offset: [usize; NDIM],
//     stride: [usize; NDIM],
//     shape: [usize; NDIM],
// ) -> (usize, usize) {
//     let start_multi_index = offset;
//     if *shape.iter().min().unwrap() == 0 {
//         // If there is a zero dimension, the start and end raw indices are the same
//         // as the subview is empty.
//         let start_raw = convert_nd_raw(start_multi_index, stride);
//         (start_raw, start_raw)
//     } else {
//         let mut end_multi_index = [0; NDIM];
//         for (index, value) in end_multi_index.iter_mut().enumerate() {
//             *value = start_multi_index[index] + shape[index] - 1;
//         }

//         let start_raw = convert_nd_raw(start_multi_index, stride);
//         let end_raw = convert_nd_raw(end_multi_index, stride);
//         // Need 1 + end_raw since `end_multi_index` is the last computed element and
//         // the range bound is one further than the last element.
//         (start_raw, 1 + end_raw)
//     }
// }

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Move the array into a subview specified by an offset and shape of the subview.
    pub fn into_subview(
        self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<ArraySubView<ArrayImpl, NDIM>, NDIM> {
        Array::new(ArraySubView::new(self, offset, shape))
    }
}
