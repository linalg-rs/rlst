//! Operations on arrays
use super::*;
use rlst_common::traits::*;

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + DefaultIterator<Item = Item>,
        const NDIM: usize,
    > FillFrom<Other> for Array<Item, ArrayImpl, NDIM>
{
    fn fill_from(&mut self, other: &Other) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item = other_item;
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + DefaultIterator<Item = Item>,
        const NDIM: usize,
    > SumInto<Other> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn sum_into(&mut self, alpha: Self::Item, other: &Other) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += alpha * other_item;
        }
    }
}
