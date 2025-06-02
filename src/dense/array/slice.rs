//! Array slicing.

use crate::{
    dense::{
        layout::convert_1d_nd_from_shape,
        number_types::{IsGreaterByOne, IsGreaterZero, NumberType},
        traits::{UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut},
    },
    BaseItem,
};

use super::{
    Array, Shape, Stride, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

/// Generic structure to store Array slices.
pub struct ArraySlice<ArrayImpl, const ADIM: usize, const NDIM: usize>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    arr: Array<ArrayImpl, ADIM>,
    // The first entry is the axis, the second is the index in the axis.
    slice: [usize; 2],
    mask: [usize; NDIM],
}

// Implementation of ArraySlice

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> BaseItem for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    ArrayImpl: BaseItem,
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> ArraySlice<ArrayImpl, ADIM, NDIM>
where
    ArrayImpl: Shape<ADIM>,
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    /// Create new array slice
    pub fn new(arr: Array<ArrayImpl, ADIM>, slice: [usize; 2]) -> Self {
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

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_value_unchecked(orig_index)
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
    ArrayImpl: UnsafeRandomAccessByRef<ADIM>,
{
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked(orig_index)
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Shape<NDIM>
    for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
    ArrayImpl: Shape<ADIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_shape = self.arr.shape();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_shape[index + self.mask[index]];
        }

        result
    }
}

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> Stride<NDIM>
    for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
    ArrayImpl: Stride<ADIM>,
{
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride: [usize; ADIM] = self.arr.stride();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_stride[index + self.mask[index]];
        }

        result
    }
}

impl<ArrayImpl, const ADIM: usize> Array<ArrayImpl, ADIM>
where
    ArrayImpl: Shape<ADIM>,
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
    ) -> Array<ArraySlice<ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsGreaterByOne<NDIM>,
        NumberType<NDIM>: IsGreaterZero,
    {
        Array::new(ArraySlice::new(self, [axis, index]))
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<ADIM>, const ADIM: usize, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked_mut(orig_index)
    }
}

impl<
        ArrayImpl: UnsafeRandomAccessByValue<ADIM> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape()))
    }
}

impl<
        ArrayImpl: UnsafeRandomAccessByRef<ADIM> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessByRef for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape()))
    }
}

impl<
        ArrayImpl: UnsafeRandomAccessMut<ADIM> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandom1DAccessMut for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape()))
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
