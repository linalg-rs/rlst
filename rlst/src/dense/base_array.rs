//! Definition of [BaseArray], a container for array data.
//!
//! A [BaseArray] is a simple container for array data. It is mainly a convient interface
//! to a data container and adds a `shape`, `stride`, and n-dimensional accessor methods.

use crate::{
    dense::layout::{check_multi_index_in_bounds, col_major_stride_from_shape},
    traits::{
        accessors::{
            RawAccess, RawAccessMut, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
            UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
            UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, ResizeInPlace, Shape, Stride},
        data_container::{
            ContainerTypeHint, DataContainer, MutableRawAccessDataContainer,
            RawAccessDataContainer, RefDataContainer, RefDataContainerMut, ResizeableDataContainer,
            ValueDataContainer,
        },
    },
};

/// Definition of a [BaseArray]. The `data` stores the actual array data, `shape` stores
/// the shape of the array, and `stride` contains the `stride` of the underlying data.
pub struct BaseArray<Data, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
}

impl<Data: ContainerTypeHint, const NDIM: usize> ContainerTypeHint for BaseArray<Data, NDIM> {
    type TypeHint = Data::TypeHint;
}

impl<Data: DataContainer, const NDIM: usize> BaseArray<Data, NDIM> {
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

impl<Data, const NDIM: usize> Shape<NDIM> for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Data: DataContainer, const NDIM: usize> BaseItem for BaseArray<Data, NDIM> {
    type Item = Data::Item;
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandom1DAccessByValue
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.data.get_unchecked_value(index)
    }
}

impl<Data: RefDataContainer, const NDIM: usize> UnsafeRandom1DAccessByRef
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.data.get_unchecked(index)
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.data.get_unchecked_mut(index)
    }
}

impl<Data: ValueDataContainer, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_value(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Data: RefDataContainerMut, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        debug_assert!(check_multi_index_in_bounds(multi_index, self.shape()));
        self.data
            .get_unchecked_mut(compute_col_major_index(multi_index, self.shape))
    }
}

impl<Data: RawAccessDataContainer, const NDIM: usize> RawAccess for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.data.data()
    }
}

impl<Data: MutableRawAccessDataContainer, const NDIM: usize> RawAccessMut
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.data_mut()
    }
}

impl<Data, const NDIM: usize> Stride<NDIM> for BaseArray<Data, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        col_major_stride_from_shape(self.shape)
    }
}

impl<Data: ResizeableDataContainer, const NDIM: usize> ResizeInPlace<NDIM>
    for BaseArray<Data, NDIM>
{
    #[inline(always)]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        let new_len = shape.iter().product();
        self.data.resize(new_len);
        self.shape = shape;
    }
}

#[inline(always)]
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
