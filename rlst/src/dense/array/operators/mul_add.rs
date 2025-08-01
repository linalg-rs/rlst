//! MulAdd operation on two arrays.
//!
//! This module provides the `MulAdd` trait, which allows for efficient multiplication and addition
//! of two arrays with a scalar.

use num::traits::MulAdd;

use crate::{
    dense::array::Array,
    traits::{
        accessors::{UnsafeRandom1DAccessByValue, UnsafeRandomAccessByValue},
        base_operations::{BaseItem, Shape},
    },
    ContainerTypeHint, ContainerTypeSelector, SelectContainerType,
};

/// Struct that represents the `mul_add` operation on two arrays.
pub struct MulAddImpl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> {
    // The first array
    arr1: Array<ArrayImpl1, NDIM>,
    arr2: Array<ArrayImpl2, NDIM>,
    scalar: Item,
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize>
    MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
{
    /// Create a new `MulAdd` instance.
    pub fn new(arr1: Array<ArrayImpl1, NDIM>, arr2: Array<ArrayImpl2, NDIM>, scalar: Item) -> Self
    where
        ArrayImpl1: BaseItem<Item = Item> + Shape<NDIM>,
        ArrayImpl2: BaseItem<Item = Item> + Shape<NDIM>,
        Item: MulAdd<Output = Item> + Copy,
    {
        // Ensure that both arrays have the same shape
        assert_eq!(
            arr1.shape(),
            arr2.shape(),
            "Arrays must have the same shape"
        );

        Self { arr1, arr2, scalar }
    }
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> ContainerTypeHint
    for MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
where
    ArrayImpl1: ContainerTypeHint,
    ArrayImpl2: ContainerTypeHint,
    SelectContainerType: ContainerTypeSelector<ArrayImpl1::TypeHint, ArrayImpl2::TypeHint>,
{
    type TypeHint = <ArrayImpl1 as ContainerTypeHint>::TypeHint;
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> BaseItem
    for MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
where
    ArrayImpl1: BaseItem<Item = Item>,
    ArrayImpl2: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> Shape<NDIM>
    for MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr1.shape()
    }
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
where
    ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item>,
    ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item>,
    Item: num::traits::MulAdd<Output = Item> + Copy,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr1
            .get_value_unchecked(multi_index)
            .mul_add(self.scalar, self.arr2.get_value_unchecked(multi_index))
    }
}

impl<ArrayImpl1, ArrayImpl2, Item, const NDIM: usize> UnsafeRandom1DAccessByValue
    for MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>
where
    ArrayImpl1: UnsafeRandom1DAccessByValue<Item = Item>,
    ArrayImpl2: UnsafeRandom1DAccessByValue<Item = Item>,
    Item: num::traits::MulAdd<Output = Item> + Copy,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr1
            .get_value_1d_unchecked(index)
            .mul_add(self.scalar, self.arr2.get_value_1d_unchecked(index))
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> MulAdd<Item, Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: BaseItem<Item = Item> + Shape<NDIM>,
    ArrayImpl2: BaseItem<Item = Item> + Shape<NDIM>,
    Item: MulAdd<Output = Item> + Copy,
{
    type Output = Array<MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>;
    /// Compentwie form `self * a + b`, where `a` is a scalar and `b` is another array.
    /// The implementation depdends on the `MulAdd` trait from the `num` crate for the componets of
    /// the arrays.
    fn mul_add(
        self,
        a: Item,
        b: Array<ArrayImpl2, NDIM>,
    ) -> Array<MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>
    where
        ArrayImpl1: BaseItem<Item = Item> + Shape<NDIM>,
        ArrayImpl2: BaseItem<Item = Item> + Shape<NDIM>,
        Item: MulAdd<Output = Item> + Copy,
    {
        Array::new(MulAddImpl::new(self, b, a))
    }
}
