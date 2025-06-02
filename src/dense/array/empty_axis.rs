//! Extend an array by an empty axis either at the front or back.

use crate::{
    dense::{
        number_types::{IsSmallerByOne, NumberType},
        traits::{UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut},
    },
    BaseItem,
};

use super::{
    Array, RawAccess, RawAccessMut, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Axis position.
#[derive(Clone, Copy)]
pub enum AxisPosition {
    /// Insert axis at the front.
    Front,
    /// Insert axis at the back.
    Back,
}

/// Array implementation that provides an appended empty axis.
pub struct ArrayAppendAxis<ArrayImpl, const ADIM: usize, const NDIM: usize>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    arr: Array<ArrayImpl, ADIM>,
    axis_position: AxisPosition,
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    /// Create new
    pub fn new(arr: Array<ArrayImpl, ADIM>, axis_position: AxisPosition) -> Self {
        Self { arr, axis_position }
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> BaseItem
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr
            .get_value_unchecked(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessByValue,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessByRef,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandom1DAccessMut,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(index)
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessByRef<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr
            .get_unchecked(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: UnsafeRandomAccessMut<ADIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr
            .get_unchecked_mut(multi_index_to_orig(multi_index, self.axis_position))
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Shape<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: Shape<ADIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [1; NDIM];
        let orig_shape = self.arr.shape();

        match self.axis_position {
            AxisPosition::Front => result[1..].copy_from_slice(&orig_shape),
            AxisPosition::Back => result[..ADIM].copy_from_slice(&orig_shape),
        }
        result
    }
}

impl<ArrayImpl: RawAccess, const ADIM: usize, const NDIM: usize> RawAccess
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

impl<ArrayImpl: RawAccessMut, const ADIM: usize, const NDIM: usize> RawAccessMut
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Stride<NDIM>
    for ArrayAppendAxis<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
    ArrayImpl: Stride<ADIM> + Shape<ADIM>,
{
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride = self.arr.stride();

        match self.axis_position {
            AxisPosition::Front => {
                result[0] = 1;
                result[1..].copy_from_slice(&orig_stride)
            }
            AxisPosition::Back => {
                result[NDIM - 1] = orig_stride[ADIM - 1] * self.arr.shape()[ADIM - 1];
                result[..ADIM].copy_from_slice(&orig_stride)
            }
        }
        result
    }
}

impl<ArrayImpl, const ADIM: usize> Array<ArrayImpl, ADIM> {
    /// Insert empty axis
    pub fn insert_empty_axis<const NDIM: usize>(
        self,
        axis_position: AxisPosition,
    ) -> Array<ArrayAppendAxis<ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsSmallerByOne<NDIM>,
    {
        Array::new(ArrayAppendAxis::new(self, axis_position))
    }
}

/// Map an extended index of dimension NDIM to an index of dimension ADIM with NDIM = ADIM + 1.
fn multi_index_to_orig<const ADIM: usize, const NDIM: usize>(
    multi_index: [usize; NDIM],
    axis_position: AxisPosition,
) -> [usize; ADIM]
where
    NumberType<ADIM>: IsSmallerByOne<NDIM>,
{
    let mut orig = [0; ADIM];
    match axis_position {
        AxisPosition::Front => orig.copy_from_slice(&multi_index[1..]),
        AxisPosition::Back => orig.copy_from_slice(&multi_index[..ADIM]),
    }

    orig
}
