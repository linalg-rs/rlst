//! Container representing multiplication with a scalar

use crate::dense::{
    array::{
        empty_chunk, Array, ChunkedAccess, DataChunk, Shape, UnsafeRandomAccessByRef,
        UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
    },
    layout::convert_1d_nd_from_shape,
    traits::{UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut},
    types::RlstNum,
};

/// Transpose array
pub struct ArrayTranspose<
    Item: RlstNum,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<Item, ArrayImpl, NDIM>,
    permutation: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayTranspose<Item, ArrayImpl, NDIM>
{
    /// Create new transpose array
    pub fn new(operator: Array<Item, ArrayImpl, NDIM>, permutation: [usize; NDIM]) -> Self {
        let mut shape = [0; NDIM];
        let operator_shape = operator.shape();

        for &index in &permutation {
            shape[index] = operator_shape[permutation[index]];
        }
        Self {
            operator,
            permutation,
            shape,
        }
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.operator
            .get_value_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.operator
            .get_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.operator
            .get_unchecked_mut(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.operator
            .get_value_unchecked(convert_1d_nd_from_shape(index, self.operator.shape()))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByRef for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.operator
            .get_unchecked(convert_1d_nd_from_shape(index, self.operator.shape()))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessMut for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.operator
            .get_unchecked_mut(convert_1d_nd_from_shape(index, self.operator.shape()))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Permute axes of an array
    pub fn permute_axes(
        self,
        permutation: [usize; NDIM],
    ) -> Array<Item, ArrayTranspose<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayTranspose::new(self, permutation))
    }

    /// Transpose an array
    pub fn transpose(self) -> Array<Item, ArrayTranspose<Item, ArrayImpl, NDIM>, NDIM> {
        let mut permutation = [0; NDIM];

        for (ind, p) in (0..NDIM).rev().enumerate() {
            permutation[ind] = p;
        }

        self.permute_axes(permutation)
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        let nelements = self.shape.iter().product();
        if let Some(mut chunk) = empty_chunk::<N, Item>(chunk_index, nelements) {
            for ind in 0..chunk.valid_entries {
                chunk.data[ind] = unsafe {
                    self.get_value_unchecked(convert_1d_nd_from_shape(
                        chunk.start_index + ind,
                        self.shape,
                    ))
                }
            }
            Some(chunk)
        } else {
            None
        }
    }
}

fn inverse_permute_multi_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    permutation: [usize; NDIM],
) -> [usize; NDIM] {
    let mut result = [0; NDIM];

    for (index, &p) in permutation.iter().enumerate() {
        result[p] = multi_index[index];
    }

    result
}
