//! Definition of [BaseArray], a container for array data.
//!
//! A [BaseArray] is a simple container for array data. It is mainly a convient interface
//! to a data container and adds a `shape`, `stride`, and n-dimensional accessor methods.

use std::iter::Copied;

use crate::dense::array::empty_chunk;
use crate::dense::data_container::{DataContainer, DataContainerMut, ResizeableDataContainerMut};
use crate::dense::layout::{
    check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw, stride_from_shape,
};
use crate::dense::traits::{
    ChunkedAccess, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandomAccessByRef,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use super::traits::ArrayIterator;
use super::types::RlstBase;

/// Definition of a [BaseArray]. The `data` stores the actual array data, `shape` stores
/// the shape of the array, and `stride` contains the `stride` of the underlying data.
pub struct BaseArray<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    BaseArray<Item, Data, NDIM>
{
    /// Create new
    pub fn new(data: Data, shape: [usize; NDIM]) -> Self {
        assert_eq!(
            data.number_of_elements(),
            shape.iter().product::<usize>(),
            "Expected {} elements but `data` has {} elements",
            shape.iter().product::<usize>(),
            data.number_of_elements()
        );

        Self { data, shape }
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> Shape<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_value(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Item: RlstBase, Data: DataContainerMut<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_mut(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Item: RlstBase, Data: DataContainerMut<Item = Item>, const N: usize, const NDIM: usize>
    ChunkedAccess<N> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        let nelements = self.shape().iter().product();
        if let Some(mut chunk) = empty_chunk(chunk_index, nelements) {
            for count in 0..chunk.valid_entries {
                unsafe {
                    chunk.data[count] = self.data.get_unchecked_value(chunk.start_index + count);
                }
            }
            Some(chunk)
        } else {
            None
        }
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> RawAccess
    for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    fn data(&self) -> &[Self::Item] {
        self.data.data()
    }

    fn offset(&self) -> usize {
        0
    }

    fn buff_ptr(&self) -> *const Self::Item {
        &self.data.data()[0]
    }
}

impl<Item: RlstBase, Data: DataContainerMut<Item = Item>, const NDIM: usize> RawAccessMut
    for BaseArray<Item, Data, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.data_mut()
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        &mut self.data.data_mut()[0]
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> Stride<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        stride_from_shape(self.shape)
    }
}

impl<Item: RlstBase, Data: ResizeableDataContainerMut<Item = Item>, const NDIM: usize>
    ResizeInPlace<NDIM> for BaseArray<Item, Data, NDIM>
{
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        let new_len = shape.iter().product();
        self.data.resize(new_len);
        self.shape = shape;
    }
}

#[inline]
fn compute_col_major_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    shape: [usize; NDIM],
) -> usize {
    let mut acc = 0;
    let mut shape_prod = 1;
    for index in 0..NDIM {
        acc += multi_index[index] * shape_prod;
        shape_prod *= shape[index];
    }
    acc
}

// impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> ArrayIterator
//     for BaseArray<Item, Data, NDIM>
// {
//     type Item = Item;

//     type Iter<'a>
//         = Copied<std::slice::Iter<'a, Item>>
//     where
//         Self: 'a;

//     fn iter(&self) -> Self::Iter<'_> {
//         self.data().iter().copied()
//     }
// }
