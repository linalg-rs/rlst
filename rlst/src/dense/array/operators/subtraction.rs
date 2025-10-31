//! Implementation of array addition

use std::ops::Sub;

use crate::{
    ContainerType, ContainerTypeSelector, SelectContainerType,
    dense::array::Array,
    traits::{
        accessors::{UnsafeRandom1DAccessByValue, UnsafeRandomAccessByValue},
        base_operations::{BaseItem, Shape},
    },
};

/// Subtraction
pub struct ArraySubtraction<ArrayImpl1, ArrayImpl2, const NDIM: usize> {
    operator1: Array<ArrayImpl1, NDIM>,
    operator2: Array<ArrayImpl2, NDIM>,
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    /// Create new
    pub fn new(operator1: Array<ArrayImpl1, NDIM>, operator2: Array<ArrayImpl2, NDIM>) -> Self {
        assert_eq!(
            operator1.shape(),
            operator2.shape(),
            "In op1 - op2 shapes not identical. op1.shape = {:#?}, op2.shape = {:#?}",
            operator1.shape(),
            operator2.shape()
        );
        Self {
            operator1,
            operator2,
        }
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> ContainerType
    for ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: ContainerType,
    ArrayImpl2: ContainerType,
    SelectContainerType: ContainerTypeSelector<ArrayImpl1::Type, ArrayImpl2::Type>,
{
    type Type =
        <SelectContainerType as ContainerTypeSelector<ArrayImpl1::Type, ArrayImpl2::Type>>::Type;
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> BaseItem
    for ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: BaseItem<Item = Item>,
    ArrayImpl2: BaseItem<Item = Item>,
    Item: Copy + Default,
{
    type Item = Item;
}

impl<Item: Copy + Default, ArrayImpl1, ArrayImpl2, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: UnsafeRandomAccessByValue<NDIM> + BaseItem<Item = Item>,
    ArrayImpl2: UnsafeRandomAccessByValue<NDIM> + BaseItem<Item = Item>,
    ArrayImpl1::Item: Sub<ArrayImpl2::Item, Output = Item>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        unsafe {
            self.operator1.get_value_unchecked(multi_index)
                - self.operator2.get_value_unchecked(multi_index)
        }
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: UnsafeRandom1DAccessByValue + BaseItem<Item = Item>,
    ArrayImpl2: UnsafeRandom1DAccessByValue + BaseItem<Item = Item>,
    ArrayImpl1::Item: Sub<ArrayImpl2::Item, Output = Item>,
    Item: Copy + Default,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        unsafe {
            self.operator1.imp().get_value_1d_unchecked(index)
                - self.operator2.imp().get_value_1d_unchecked(index)
        }
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> Shape<NDIM>
    for ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.operator1.shape()
    }
}
