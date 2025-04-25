//! Default view onto an array.

use crate::dense::array::Array;

use crate::dense::traits::{
    RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandom1DAccessByRef,
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Basic structure for a `View`
pub struct ArrayView<'a, ArrayImpl, const NDIM: usize>(&'a Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ArrayView<'a, ArrayImpl, NDIM> {
    /// Create new view
    #[inline(always)]
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayView<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArrayView<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayView<'_, ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayView<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

/////////// ArrayViewMut

/// Mutable array view
pub struct ArrayViewMut<'a, ArrayImpl, const NDIM: usize>(&'a mut Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ArrayViewMut<'a, ArrayImpl, NDIM> {
    /// Create new mutable view
    #[inline(always)]
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayViewMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(index)
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM>
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayViewMut<'_, ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl: ResizeInPlace<NDIM>, const NDIM: usize> ResizeInPlace<NDIM>
    for ArrayViewMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}
