//! Container representing multiplication with a scalar

use crate::array::*;

pub struct ArrayConjugate<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<Item, ArrayImpl, NDIM>,
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayConjugate<Item, ArrayImpl, NDIM>
{
    pub fn new(operator: Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { operator }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayConjugate<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.operator.get_value_unchecked(indices).conj()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayConjugate<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn conj(self) -> Array<Item, ArrayConjugate<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayConjugate::new(self))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayConjugate<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        if let Some(mut chunk) = self.operator.get_chunk(chunk_index) {
            for item in &mut chunk.data {
                *item = item.conj();
            }
            Some(chunk)
        } else {
            None
        }
    }
}
