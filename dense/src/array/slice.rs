//! Array slicing.

use crate::{
    layout::convert_1d_nd_from_shape,
    number_types::{IsGreaterByOne, IsGreaterZero, NumberType},
};

use super::*;

/// Generic structure to store Array slices.
pub struct ArraySlice<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
    const ADIM: usize,
    const NDIM: usize,
> where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    arr: Array<Item, ArrayImpl, ADIM>,
    // The first entry is the axis, the second is the index in the axis.
    slice: [usize; 2],
    mask: [usize; NDIM],
}

// Implementation of ArraySlice

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    pub fn new(arr: Array<Item, ArrayImpl, ADIM>, slice: [usize; 2]) -> Self {
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

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_value_unchecked(orig_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessByRef<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked(orig_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Shape<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn shape(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_shape = self.arr.shape();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_shape[index + self.mask[index]];
        }

        result
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccess<Item = Item>
            + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccess for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;
    fn data(&self) -> &[Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) =
            compute_raw_range(self.slice, self.arr.stride(), self.arr.shape());

        &self.arr.data()[start_raw..end_raw]
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM> + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
    > Stride<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn stride(&self) -> [usize; NDIM] {
        let mut result = [0; NDIM];
        let orig_stride: [usize; ADIM] = self.arr.stride();

        for (index, value) in result.iter_mut().enumerate() {
            *value = orig_stride[index + self.mask[index]];
        }

        result
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM>,
        const ADIM: usize,
    > Array<Item, ArrayImpl, ADIM>
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
    ) -> Array<Item, ArraySlice<Item, ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsGreaterByOne<NDIM>,
        NumberType<NDIM>: IsGreaterZero,
    {
        Array::new(ArraySlice::new(self, [axis, index]))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item> + Shape<ADIM> + Stride<ADIM>,
        const ADIM: usize,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<rlst_common::types::DataChunk<Self::Item, N>> {
        let nelements = self.shape().iter().product();
        if let Some(mut chunk) = empty_chunk(chunk_index, nelements) {
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

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + UnsafeRandomAccessByRef<ADIM, Item = Item>
            + UnsafeRandomAccessMut<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    type Item = Item;

    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        let mut orig_index = multi_index_to_orig(multi_index, self.mask);
        orig_index[self.slice[0]] = self.slice[1];
        self.arr.get_unchecked_mut(orig_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<ADIM, Item = Item>
            + Shape<ADIM>
            + RawAccessMut<Item = Item>
            + Stride<ADIM>
            + UnsafeRandomAccessMut<ADIM, Item = Item>,
        const ADIM: usize,
        const NDIM: usize,
    > RawAccessMut for ArraySlice<Item, ArrayImpl, ADIM, NDIM>
where
    NumberType<ADIM>: IsGreaterByOne<NDIM>,
    NumberType<NDIM>: IsGreaterZero,
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        assert!(!self.is_empty());
        let (start_raw, end_raw) =
            compute_raw_range(self.slice, self.arr.stride(), self.arr.shape());
        &mut self.arr.data_mut()[start_raw..end_raw]
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

fn compute_raw_range<const NDIM: usize>(
    slice: [usize; 2],
    stride: [usize; NDIM],
    shape: [usize; NDIM],
) -> (usize, usize) {
    use crate::layout::convert_nd_raw;
    let mut start_multi_index = [0; NDIM];
    start_multi_index[slice[0]] = slice[1];
    let mut end_multi_index = shape;
    for (index, value) in end_multi_index.iter_mut().enumerate() {
        if index == slice[0] {
            *value = slice[1]
        } else {
            // We started with the shape. Reduce
            // each value of the shape by 1 to get last
            // index in that dimension.
            assert!(*value > 0);
            *value -= 1;
        }
    }
    (
        convert_nd_raw(start_multi_index, stride),
        1 + convert_nd_raw(end_multi_index, stride),
    )
}

#[cfg(test)]
mod test {

    use crate::traits::*;
    use crate::{layout::convert_nd_raw, rlst_dynamic_array3};

    #[test]
    fn test_create_slice() {
        let shape = [3, 7, 6];
        let mut arr = rlst_dynamic_array3!(f64, shape);

        arr.fill_from_seed_equally_distributed(0);

        let slice = arr.view().slice(1, 2);

        assert_eq!(slice[[1, 5]], arr[[1, 2, 5]]);

        assert_eq!(slice.shape(), [3, 6]);

        let stride_expected = [arr.stride()[0], arr.stride()[2]];
        let stride_actual = slice.stride();

        assert_eq!(stride_expected, stride_actual);

        let orig_data = arr.data();
        let slice_data = slice.data();

        let orig_raw_index = convert_nd_raw([1, 2, 5], arr.stride());
        let slice_raw_index = convert_nd_raw([1, 5], slice.stride());

        assert_eq!(orig_data[orig_raw_index], slice_data[slice_raw_index]);
        assert_eq!(slice_data[slice_raw_index], slice[[1, 5]]);

        let last_raw_index = convert_nd_raw([2, 5], slice.stride());
        assert_eq!(slice_data[last_raw_index], slice[[2, 5]]);
    }

    #[test]
    fn test_multiple_slices() {
        let shape = [3, 7, 6];
        let mut arr = rlst_dynamic_array3!(f64, shape);
        arr.fill_from_seed_equally_distributed(0);

        let mut slice = arr.view_mut().slice(1, 3).slice(1, 1);

        slice[[2]] = 5.0;

        assert_eq!(slice.shape(), [3]);
        assert_eq!(arr[[2, 3, 1]], 5.0);
    }

    #[test]
    fn test_slice_of_subview() {
        let shape = [3, 7, 6];
        let mut arr = rlst_dynamic_array3!(f64, shape);
        arr.fill_from_seed_equally_distributed(0);
        let mut arr2 = rlst_dynamic_array3!(f64, shape);
        arr2.fill_from(arr.view());

        let slice = arr.into_subview([1, 2, 4], [2, 3, 2]).slice(1, 2);

        assert_eq!(slice.shape(), [2, 2]);

        assert_eq!(slice[[1, 0]], arr2[[2, 4, 4]]);

        let slice_data = slice.data();
        let arr_data = arr2.data();

        let slice_index = convert_nd_raw([1, 0], slice.stride());
        let array_index = convert_nd_raw([2, 4, 4], arr2.stride());

        assert_eq!(slice_data[slice_index], arr_data[array_index]);
    }
}
