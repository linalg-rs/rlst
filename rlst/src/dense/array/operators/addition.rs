//! Implementation of array addition

use std::ops::Add;

use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, base_operations::BaseItem},
    ContainerType, ContainerTypeSelector, SelectContainerType,
};

/// Addition
pub struct ArrayAddition<ArrayImpl1, ArrayImpl2, const NDIM: usize> {
    operator1: Array<ArrayImpl1, NDIM>,
    operator2: Array<ArrayImpl2, NDIM>,
}

impl<ArrayImpl1: Shape<NDIM>, ArrayImpl2: Shape<NDIM>, const NDIM: usize>
    ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
{
    /// Create new
    pub fn new(operator1: Array<ArrayImpl1, NDIM>, operator2: Array<ArrayImpl2, NDIM>) -> Self {
        assert_eq!(
            operator1.shape(),
            operator2.shape(),
            "In op1 + op2 shapes not identical. op1.shape = {:#?}, op2.shape = {:#?}",
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
    for ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1: ContainerType,
    ArrayImpl2: ContainerType,
    SelectContainerType: ContainerTypeSelector<ArrayImpl1::Type, ArrayImpl2::Type>,
{
    type Type =
        <SelectContainerType as ContainerTypeSelector<ArrayImpl1::Type, ArrayImpl2::Type>>::Type;
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> BaseItem
    for ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
where
    Item: Copy + Default,
    ArrayImpl1: BaseItem<Item = Item>,
    ArrayImpl2: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<
        Item,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM> + BaseItem<Item = Item>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM> + BaseItem<Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1::Item: Add<ArrayImpl2::Item, Output = Item>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.operator1.get_value_unchecked(multi_index)
            + self.operator2.get_value_unchecked(multi_index)
    }
}

impl<
        Item,
        ArrayImpl1: UnsafeRandom1DAccessByValue + BaseItem<Item = Item>,
        ArrayImpl2: UnsafeRandom1DAccessByValue + BaseItem<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
where
    ArrayImpl1::Item: Add<ArrayImpl2::Item, Output = Item>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.operator1.get_value_1d_unchecked(index) + self.operator2.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl1: Shape<NDIM>, ArrayImpl2, const NDIM: usize> Shape<NDIM>
    for ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator1.shape()
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Add<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>: UnsafeRandomAccessByValue<NDIM>,
{
    type Output = Array<ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn add(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(ArrayAddition {
            operator1: self,
            operator2: rhs,
        })
    }
}
