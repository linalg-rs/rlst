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
pub struct ArrayFlatView<ArrayImpl, const NDIM: usize>(Array<ArrayImpl, NDIM>);

impl<ArrayImpl, const NDIM: usize> ArrayFlatView<ArrayImpl, NDIM> {
    /// Instantiate a new flatted view.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<1> for ArrayFlatView<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; 1] {
        [self.0.shape().iter().product()]
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandomAccessByValue<1>
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; 1]) -> Self::Item {
        self.0.get_value_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandomAccessByRef<1>
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; 1]) -> &Self::Item {
        self.0.get_1d_unchecked(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandomAccessMut<1>
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; 1]) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(multi_index[0])
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayFlatView<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(index)
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayFlatView<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for ArrayFlatView<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}
