//! Reverse a single axis of an array.

use crate::dense::array::Array;
use crate::dense::layout::convert_1d_nd_from_shape;
use crate::traits::accessors::{
    UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::traits::array::{BaseItem, Shape};
use crate::ContainerTypeHint;

/// This struct represents an array implementation with a single axis reversed.
pub struct ReverseAxis<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    axis: usize,
}

impl<ArrayImpl, const NDIM: usize> ReverseAxis<ArrayImpl, NDIM> {
    /// Create a new reverse axis array
    pub fn new(arr: Array<ArrayImpl, NDIM>, axis: usize) -> Self {
        Self { arr, axis }
    }
}

impl<ArrayImpl, const NDIM: usize> Shape<NDIM> for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<ArrayImpl, const NDIM: usize> ContainerTypeHint for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerTypeHint,
{
    type TypeHint = ArrayImpl::TypeHint;
}

impl<Item, ArrayImpl, const NDIM: usize> BaseItem for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM> for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(reverse_multi_index(
            multi_index,
            self.axis,
            self.arr.shape(),
        ))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByRef<NDIM> for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(reverse_multi_index(
            multi_index,
            self.axis,
            self.arr.shape(),
        ))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessMut<NDIM> for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr.get_unchecked_mut(reverse_multi_index(
            multi_index,
            self.axis,
            self.arr.shape(),
        ))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        let multi_index = convert_1d_nd_from_shape(index, self.arr.shape());
        self.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByRef for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        let multi_index = convert_1d_nd_from_shape(index, self.arr.shape());
        self.get_unchecked(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessMut for ReverseAxis<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        let multi_index = convert_1d_nd_from_shape(index, self.arr.shape());
        self.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Reverse a single axis of the array.
    pub fn reverse_axis(self, axis: usize) -> Array<ReverseAxis<ArrayImpl, NDIM>, NDIM> {
        assert!(axis < NDIM, "Axis out of bounds");
        Array::new(ReverseAxis::new(self, axis))
    }
}

#[inline(always)]
fn reverse_multi_index<const NDIM: usize>(
    mut multi_index: [usize; NDIM],
    axis: usize,
    shape: [usize; NDIM],
) -> [usize; NDIM] {
    multi_index[axis] = shape[axis] - 1 - multi_index[axis];
    multi_index
}
