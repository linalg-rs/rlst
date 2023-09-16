use crate::data_container::{DataContainer, DataContainerMut};
use crate::layout::{convert_nd_1d, stride_from_shape};
use rlst_common::traits::{UnsafeRandomAccessByValue, UnsafeRandomAccessMut};
use rlst_common::{
    traits::{Shape, UnsafeRandomAccessByRef},
    types::Scalar,
};

pub struct BaseArray<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize> {
    data: Data,
    shape: [usize; NDIM],
    stride: [usize; NDIM],
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    BaseArray<Item, Data, NDIM>
{
    pub fn new(data: Data, shape: [usize; NDIM]) -> Self {
        let stride = stride_from_shape(shape);
        Self {
            data,
            shape,
            stride,
        }
    }

    pub fn new_with_stride(data: Data, shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        Self {
            data,
            shape,
            stride,
        }
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize> Shape<NDIM>
    for BaseArray<Item, Data, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByRef<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    unsafe fn get_unchecked(&self, indices: [usize; NDIM]) -> &Self::Item {
        let index = convert_nd_1d(indices, self.stride);
        self.data.get_unchecked(index)
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        let index = convert_nd_1d(indices, self.stride);
        self.data.get_unchecked_value(index)
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, const NDIM: usize>
    UnsafeRandomAccessMut<NDIM> for BaseArray<Item, Data, NDIM>
{
    type Item = Item;

    unsafe fn get_unchecked_mut(&mut self, indices: [usize; NDIM]) -> &mut Self::Item {
        let index = convert_nd_1d(indices, self.stride);
        self.data.get_unchecked_mut(index)
    }
}
