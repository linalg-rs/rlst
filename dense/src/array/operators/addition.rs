//! Implementation of array addition

use crate::array::*;

pub struct ArrayAddition<
    Item: Scalar,
    ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator1: Array<Item, ArrayImpl1, NDIM>,
    operator2: Array<Item, ArrayImpl2, NDIM>,
}

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayAddition<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    pub fn new(
        operator1: Array<Item, ArrayImpl1, NDIM>,
        operator2: Array<Item, ArrayImpl2, NDIM>,
    ) -> Self {
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

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayAddition<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.operator1.get_value_unchecked(indices) + self.operator2.get_value_unchecked(indices)
    }
}

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayAddition<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator1.shape()
    }
}

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > std::ops::Add<Array<Item, ArrayImpl2, NDIM>> for Array<Item, ArrayImpl1, NDIM>
{
    type Output = Array<Item, ArrayAddition<Item, ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn add(self, rhs: Array<Item, ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(ArrayAddition {
            operator1: self,
            operator2: rhs,
        })
    }
}
