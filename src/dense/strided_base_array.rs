//! Definition of [StridedBaseArray], a container for array data with custom stride.
//!
//! A [StridedBaseArray] is a simple container for array data. It is mainly a convient interface
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

use super::traits::{
    ArrayIterator, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
};
use super::types::RlstBase;

/// Definition of a [StridedBaseArray]. The `data` stores the actual array data, `shape` stores
/// the shape of the array, and `stride` contains the `stride` of the underlying data.
pub struct StridedBaseArray<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    StridedBaseArray<Item, Data, NDIM>
{
    /// Create new
    pub fn new(data: Data, shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        assert_eq!(
            data.number_of_elements(),
            shape.iter().product::<usize>(),
            "Expected {} elements but `data` has {} elements",
            shape.iter().product::<usize>(),
            data.number_of_elements()
        );

        Self {
            data,
            shape,
            stride,
        }
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> Shape<NDIM>
    for StridedBaseArray<Item, Data, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_value(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandom1DAccessByValue for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> UnsafeRandom1DAccessByRef
    for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<
        Item: RlstBase,
        Data: DataContainer<Item = Item> + DataContainerMut<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessMut for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<Item: RlstBase, Data: DataContainerMut<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for StridedBaseArray<Item, Data, NDIM>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_mut(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Item: RlstBase, Data: DataContainerMut<Item = Item>, const N: usize, const NDIM: usize>
    ChunkedAccess<N> for StridedBaseArray<Item, Data, NDIM>
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
                    chunk.data[count] = self.get_value_unchecked(convert_1d_nd_from_shape(
                        chunk.start_index + count,
                        self.shape,
                    ));
                }
            }
            Some(chunk)
        } else {
            None
        }
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> RawAccess
    for StridedBaseArray<Item, Data, NDIM>
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
    for StridedBaseArray<Item, Data, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.data_mut()
    }

    fn buff_ptr_mut(&mut self) -> *mut Self::Item {
        &mut self.data.data_mut()[0]
    }
}

impl<Item: RlstBase, Data: DataContainer<Item = Item>, const NDIM: usize> Stride<NDIM>
    for StridedBaseArray<Item, Data, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.stride
    }
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
