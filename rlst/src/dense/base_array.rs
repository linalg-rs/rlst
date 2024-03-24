//! Definition of [BaseArray], a container for array data.
//!
//! A [BaseArray] is a simple container for array data. It is mainly a convient interface
//! to a data container and adds a `shape`, `stride`, and n-dimensional accessor methods.

use crate::dense::array::empty_chunk;
use crate::dense::data_container::{DataContainer, DataContainerMut, ResizeableDataContainerMut};
use crate::dense::layout::{
    check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw, stride_from_shape,
};
use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::RlstScalar;

/// Definition of a [BaseArray]. The `data` stores the actual array data, `shape` stores
/// the shape of the array, and `stride` contains the `stride` of the underlying data.
pub struct BaseArray<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    BaseArray<Item, Data, NDIM>
{
    /// Create new
    pub fn new(data: Data, shape: [usize; NDIM]) -> Self {
        let stride = stride_from_shape(shape);
        Self::new_with_stride(data, shape, stride)
    }

    /// Create new with stride
    pub fn new_with_stride(data: Data, shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        if *shape.iter().min().unwrap() == 0 {
            // Array is empty
            assert_eq!(
                data.number_of_elements(),
                0,
                "Expected 0 elements but `data` has {} elements",
                data.number_of_elements()
            );
        } else {
            // Array is not empty
            let mut largest_index = [0; NDIM];
            largest_index.copy_from_slice(&shape.iter().map(|elem| elem - 1).collect::<Vec<_>>());
            let raw_index = convert_nd_raw(largest_index, stride);
            assert_eq!(
                1 + raw_index,
                data.number_of_elements(),
                "`data` has {} elements but expected {} elements from shape and stride.",
                data.number_of_elements(),
                1 + raw_index
            );
        }

        Self {
            data,
            shape,
            stride,
        }
    }
}

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize> Shape<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked(index)
    }
}

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked_value(index)
    }
}

impl<Item: RlstScalar, Data: DataContainerMut<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        let index = convert_nd_raw(multi_index, self.stride);
        self.data.get_unchecked_mut(index)
    }
}

impl<Item: RlstScalar, Data: DataContainerMut<Item = Item>, const N: usize, const NDIM: usize>
    ChunkedAccess<N> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
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

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize> RawAccess
    for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    fn data(&self) -> &[Self::Item] {
        self.data.data()
    }
}

impl<Item: RlstScalar, Data: DataContainerMut<Item = Item>, const NDIM: usize> RawAccessMut
    for BaseArray<Item, Data, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.data_mut()
    }
}

impl<Item: RlstScalar, Data: DataContainer<Item = Item>, const NDIM: usize> Stride<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.stride
    }
}

impl<Item: RlstScalar, Data: ResizeableDataContainerMut<Item = Item>, const NDIM: usize>
    ResizeInPlace<NDIM> for BaseArray<Item, Data, NDIM>
{
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        let new_len = shape.iter().product();
        self.data.resize(new_len);
        self.stride = stride_from_shape(shape);
        self.shape = shape;
    }
}
