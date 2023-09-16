//! Basic Array type

use rlst_common::traits::Shape;
use rlst_common::traits::UnsafeRandomAccessByRef;
use rlst_common::traits::UnsafeRandomAccessByValue;
use rlst_common::traits::UnsafeRandomAccessMut;
use rlst_common::types::Scalar;

pub mod iterators;

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
    unsafe fn get_unchecked_mut(&mut self, indices: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(indices)
    }
}
