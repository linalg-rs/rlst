//! Array slicing.

use crate::{
    ContainerType, MemoryLayout, RawAccess, RawAccessMut, Unknown,
    base_types::NumberType,
    dense::layout::convert_1d_nd_from_shape,
    traits::{
        accessors::{
            UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
        },
        base_operations::BaseItem,
        number_relations::{IsGreaterByOne, IsGreaterZero},
    },
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

impl<ArrayImpl, const ADIM: usize, const NDIM: usize> ContainerType
    for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    ArrayImpl: ContainerType,
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Type = Unknown;
}

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
        unsafe { self.arr.get_value_unchecked(orig_index) }
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
        unsafe { self.arr.get_unchecked(orig_index) }
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
        unsafe { self.arr.get_unchecked_mut(orig_index) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<ADIM> + Shape<ADIM>, const ADIM: usize, const NDIM: usize>
    UnsafeRandom1DAccessByValue for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe { self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<ADIM> + Shape<ADIM>, const ADIM: usize, const NDIM: usize>
    UnsafeRandom1DAccessByRef for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        unsafe { self.get_unchecked(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<ADIM> + Shape<ADIM>, const ADIM: usize, const NDIM: usize>
    UnsafeRandom1DAccessMut for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        unsafe { self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape())) }
    }
}

impl<
    Item,
    ArrayImpl: Shape<ADIM> + RawAccess<Item = Item> + Stride<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> RawAccess for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn data(&self) -> Option<&[Self::Item]> {
        if self.slice[0] == 0 && matches!(self.arr.memory_layout(), MemoryLayout::RowMajor) {
            // The layout is row-major and we are slicing at the left-most dimension
            let step = *self.arr.stride().first().unwrap();
            Some(&self.arr.data().unwrap()[self.slice[1] * step..(1 + self.slice[1]) * step])
        } else if self.slice[0] == ADIM - 1
            && matches!(self.arr.memory_layout(), MemoryLayout::ColumnMajor)
        {
            // The layout is column-major and we are slicing at the right-most dimension
            let step = *self.arr.stride().last().unwrap();
            Some(&self.arr.data().unwrap()[self.slice[1] * step..(1 + self.slice[1]) * step])
        } else {
            None
        }
    }
}

impl<
    Item,
    ArrayImpl: Shape<ADIM> + RawAccessMut<Item = Item> + Stride<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> RawAccessMut for ArraySlice<ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn data_mut(&mut self) -> Option<&mut [Self::Item]> {
        if self.slice[0] == 0 && matches!(self.arr.memory_layout(), MemoryLayout::RowMajor) {
            // The layout is row-major and we are slicing at the left-most dimension
            let step = *self.arr.stride().first().unwrap();
            Some(
                &mut self.arr.data_mut().unwrap()[self.slice[1] * step..(1 + self.slice[1]) * step],
            )
        } else if self.slice[0] == ADIM - 1
            && matches!(self.arr.memory_layout(), MemoryLayout::ColumnMajor)
        {
            // The layout is column-major and we are slicing at the right-most dimension
            let step = *self.arr.stride().last().unwrap();
            Some(
                &mut self.arr.data_mut().unwrap()[self.slice[1] * step..(1 + self.slice[1]) * step],
            )
        } else {
            None
        }
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

#[cfg(test)]
mod test {

    use crate::DynArray;
    use crate::StridedDynArray;

    #[test]
    fn test_row_major_data() {
        let mut mat = StridedDynArray::<f64, _>::row_major([3, 5]);
        mat.fill_from_seed_equally_distributed(0);

        let slice = mat.r().row(1);

        let data = slice.data().unwrap();

        assert_eq!(data.len(), 5);

        for (index, &actual) in data.iter().enumerate() {
            assert_eq!(actual, slice[[index]]);
            assert_eq!(actual, mat[[1, index]]);
        }
    }

    #[test]
    fn test_col_major_data() {
        let mut mat = DynArray::<f64, 2>::from_shape([3, 5]);
        mat.fill_from_seed_equally_distributed(0);

        let slice = mat.r().col(1);

        let data = slice.data().unwrap();

        assert_eq!(data.len(), 3);

        for (index, &actual) in data.iter().enumerate() {
            assert_eq!(actual, slice[[index]]);
            assert_eq!(actual, mat[[index, 1]]);
        }
    }
}
