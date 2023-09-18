//! Basic Array type

use crate::base_array::BaseArray;
use crate::data_container::VectorContainer;
use crate::layout::convert_1d_nd;
use rlst_common::traits::RandomAccessByRef;
use rlst_common::traits::RandomAccessMut;
use rlst_common::traits::Shape;
use rlst_common::traits::UnsafeRandomAccessByRef;
use rlst_common::traits::UnsafeRandomAccessByValue;
use rlst_common::traits::UnsafeRandomAccessMut;
use rlst_common::types::Scalar;

pub mod iterators;
pub mod operations;
pub mod operators;
pub mod random;
pub mod views;

pub type DynamicArray<Item, const NDIM: usize> =
    Array<Item, BaseArray<Item, VectorContainer<Item>, NDIM>, NDIM>;

/// The basic tuple type defining an array.
pub struct Array<Item, ArrayImpl, const NDIM: usize>(ArrayImpl)
where
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>;

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn new(arr: ArrayImpl) -> Self {
        Self(arr)
    }

    pub fn new_dynamic_like_self(&self) -> DynamicArray<Item, NDIM> {
        DynamicArray::<Item, NDIM>::from_shape(self.shape())
    }
}

impl<Item: Scalar, const NDIM: usize> DynamicArray<Item, NDIM> {
    pub fn from_shape(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(BaseArray::new(VectorContainer::new(size), shape))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(indices)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, indices: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(indices)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, indices: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(indices)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > std::ops::Index<[usize; NDIM]> for Array<Item, ArrayImpl, NDIM>
{
    type Output = Item;
    #[inline]
    fn index(&self, index: [usize; NDIM]) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > std::ops::IndexMut<[usize; NDIM]> for Array<Item, ArrayImpl, NDIM>
{
    #[inline]
    fn index_mut(&mut self, index: [usize; NDIM]) -> &mut Self::Output {
        self.0.get_mut(index).unwrap()
    }
}
