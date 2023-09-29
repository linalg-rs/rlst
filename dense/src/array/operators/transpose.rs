//! Container representing multiplication with a scalar

use crate::{array::*, layout::convert_1d_nd_from_shape};

pub struct ArrayTranspose<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<Item, ArrayImpl, NDIM>,
    permutation: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayTranspose<Item, ArrayImpl, NDIM>
{
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
        Item: Scalar,
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
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayTranspose<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn permute_axes(
        self,
        permutation: [usize; NDIM],
    ) -> Array<Item, ArrayTranspose<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayTranspose::new(self, permutation))
    }

    pub fn transpose(self) -> Array<Item, ArrayTranspose<Item, ArrayImpl, NDIM>, NDIM> {
        let mut permutation = [0; NDIM];

        for (ind, p) in (0..NDIM).rev().enumerate() {
            permutation[ind] = p;
        }

        self.permute_axes(permutation)
    }
}

impl<
        Item: Scalar,
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
