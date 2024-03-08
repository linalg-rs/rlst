//! Implementation of array addition

use crate::array::*;

pub struct CmpWiseDivision<
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
    > CmpWiseDivision<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    pub fn new(
        operator1: Array<Item, ArrayImpl1, NDIM>,
        operator2: Array<Item, ArrayImpl2, NDIM>,
    ) -> Self {
        assert_eq!(
            operator1.shape(),
            operator2.shape(),
            "In op1 / op2 shapes not identical. op1.shape = {:#?}, op2.shape = {:#?}",
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
    > UnsafeRandomAccessByValue<NDIM> for CmpWiseDivision<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.operator1.get_value_unchecked(multi_index)
            / self.operator2.get_value_unchecked(multi_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for CmpWiseDivision<Item, ArrayImpl1, ArrayImpl2, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<crate::types::DataChunk<Self::Item, N>> {
        if let (Some(mut chunk1), Some(chunk2)) = (
            self.operator1.get_chunk(chunk_index),
            self.operator2.get_chunk(chunk_index),
        ) {
            for (elem1, &elem2) in chunk1.data.iter_mut().zip(chunk2.data.iter()) {
                *elem1 /= elem2;
            }
            Some(chunk1)
        } else {
            None
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl1: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        ArrayImpl2: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for CmpWiseDivision<Item, ArrayImpl1, ArrayImpl2, NDIM>
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
    > std::ops::Div<Array<Item, ArrayImpl2, NDIM>> for Array<Item, ArrayImpl1, NDIM>
{
    type Output = Array<Item, CmpWiseDivision<Item, ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn div(self, rhs: Array<Item, ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(CmpWiseDivision {
            operator1: self,
            operator2: rhs,
        })
    }
}
