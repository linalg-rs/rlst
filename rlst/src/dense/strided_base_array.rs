//! Definition of [StridedBaseArray], a container for array data with custom stride.
//!
//! A [StridedBaseArray] is a simple container for array data. It is mainly a convient interface
//! to a data container and adds a `shape`, `stride`, and n-dimensional accessor methods.

use crate::{
    dense::layout::{check_multi_index_in_bounds, convert_1d_nd_from_shape, convert_nd_raw},
    traits::{
        accessors::{
            RawAccess, RawAccessMut, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
            UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
            UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape, Stride},
        data_container::{
            ContainerType, DataContainer, MutableRawAccessDataContainer, RawAccessDataContainer,
            RefDataContainer, RefDataContainerMut, ValueDataContainer,
        },
    },
};

/// Definition of a [StridedBaseArray]. The `data` stores the actual array data, `shape` stores
/// the shape of the array, and `stride` contains the `stride` of the underlying data.
pub struct StridedBaseArray<Data, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Data: ContainerType, const NDIM: usize> ContainerType for StridedBaseArray<Data, NDIM> {
    type Type = Data::Type;
}

impl<Data: DataContainer, const NDIM: usize> StridedBaseArray<Data, NDIM> {
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

impl<Data: DataContainer, const NDIM: usize> BaseItem for StridedBaseArray<Data, NDIM> {
    type Item = Data::Item;
}

impl<Data: DataContainer, const NDIM: usize> Shape<NDIM> for StridedBaseArray<Data, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_value(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandom1DAccessByValue
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandom1DAccessByRef
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape))
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for StridedBaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_mut(convert_nd_raw(multi_index, self.stride))
    }
}

impl<Data: RawAccessDataContainer, const NDIM: usize> RawAccess for StridedBaseArray<Data, NDIM> {
    fn data(&self) -> &[Self::Item] {
        self.data.data()
    }
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> RawAccessMut
    for StridedBaseArray<Data, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.data_mut()
    }
}

impl<Data, const NDIM: usize> Stride<NDIM> for StridedBaseArray<Data, NDIM> {
    fn stride(&self) -> [usize; NDIM] {
        self.stride
    }
}
