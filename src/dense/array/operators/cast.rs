//! Cast an array to another type

use std::marker::PhantomData;

use crate::dense::{
    array::{Array, ChunkedAccess, DataChunk, Shape, UnsafeRandomAccessByValue},
    types::RlstNum,
};

/// Array to complex
pub struct ArrayCast<
    Item: RlstNum,
    Target: RlstNum,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<Item, ArrayImpl, NDIM>,
    _marker1: PhantomData<Item>,
    _marker2: PhantomData<Target>,
}

impl<
        Item: RlstNum,
        Target: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayCast<Item, Target, ArrayImpl, NDIM>
{
    /// Create new
    pub fn new(operator: Array<Item, ArrayImpl, NDIM>) -> Self {
        Self {
            operator,
            _marker1: PhantomData,
            _marker2: PhantomData,
        }
    }
}

impl<
        Item: RlstNum,
        Target: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayCast<Item, Target, ArrayImpl, NDIM>
{
    type Item = Target;
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        num::cast::<Item, Target>(self.operator.get_value_unchecked(multi_index)).unwrap()
    }
}

impl<
        Item: RlstNum,
        Target: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayCast<Item, Target, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

impl<
        Item: RlstNum,
        Target: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayCast<Item, Target, ArrayImpl, NDIM>
{
    type Item = Target;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        if let Some(chunk) = self.operator.get_chunk(chunk_index) {
            let mut data = [<Target as num::Zero>::zero(); N];

            for (d, &c) in data.iter_mut().zip(chunk.data.iter()) {
                *d = num::cast::<Item, Target>(c).unwrap();
            }
            Some(DataChunk::<Target, N> {
                data,
                start_index: chunk.start_index,
                valid_entries: chunk.valid_entries,
            })
        } else {
            None
        }
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Cast array to type `T`.
    pub fn cast<T: RlstNum>(self) -> Array<T, ArrayCast<Item, T, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayCast::new(self))
    }
}
