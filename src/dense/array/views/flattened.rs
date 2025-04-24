//! Flattened view onto an array

use crate::dense::array::Array;

use crate::dense::traits::{
    RawAccess, RawAccessMut, Shape, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
    UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};

/// A flattened view onto an array.
///
/// Use the funtion [arr.flattened()](crate::Array::view_flat) instead.
pub struct ArrayFlatView<'a, ArrayImpl, const NDIM: usize>(&'a Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ArrayFlatView<'a, ArrayImpl, NDIM> {
    /// Instantiate a new flatted view.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<1> for ArrayFlatView<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; 1] {
        [self.0.shape().iter().product()]
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandomAccessByValue<1>
    for ArrayFlatView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        self.0.get_value_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandomAccessByRef<1>
    for ArrayFlatView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        self.0.get_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayFlatView<'_, ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

////////// Mutable flattened view

/// A flattened view onto an array.
///
/// Use the funtion [arr.flattened_mut()](crate::Array::view_flat_mut) instead.
pub struct ArrayFlatViewMut<'a, ArrayImpl, const NDIM: usize>(&'a mut Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ArrayFlatViewMut<'a, ArrayImpl, NDIM> {
    /// Instantiate a new flattened view.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<1> for ArrayFlatViewMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; 1] {
        [self.0.shape().iter().product()]
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandomAccessByValue<1>
    for ArrayFlatViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        self.0.get_value_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandomAccessMut<1>
    for ArrayFlatViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; 1]) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandomAccessByRef<1>
    for ArrayFlatViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        self.0.get_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayFlatViewMut<'_, ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut
    for ArrayFlatViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }
}
