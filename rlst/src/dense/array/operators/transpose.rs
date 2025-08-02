//! Container representing multiplication with a scalar

use crate::{
    dense::{array::Array, layout::convert_1d_nd_from_shape},
    traits::{
        accessors::{
            UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
            UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape},
    },
    ContainerType,
};

/// Transpose array
pub struct ArrayTranspose<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    permutation: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<ArrayImpl, const NDIM: usize> ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create new transpose array
    pub fn new(arr: Array<ArrayImpl, NDIM>, permutation: [usize; NDIM]) -> Self {
        let mut shape = [0; NDIM];
        let operator_shape = arr.shape();

        for &index in &permutation {
            shape[index] = operator_shape[permutation[index]];
        }
        Self {
            arr,
            permutation,
            shape,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> ContainerType for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<Item, ArrayImpl, const NDIM: usize> BaseItem for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
{
    type Item = Item;
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr
            .get_value_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByRef<NDIM> for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr
            .get_unchecked(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessMut<NDIM> for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr
            .get_unchecked_mut(inverse_permute_multi_index(multi_index, self.permutation))
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayTranspose<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.get_value_unchecked(convert_1d_nd_from_shape(index, self.shape()))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByRef for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.get_unchecked(convert_1d_nd_from_shape(index, self.shape()))
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessMut for ArrayTranspose<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.get_unchecked_mut(convert_1d_nd_from_shape(index, self.shape()))
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Permute axes of an array
    pub fn permute_axes(
        self,
        permutation: [usize; NDIM],
    ) -> Array<ArrayTranspose<ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayTranspose::new(self, permutation))
    }

    /// Transpose an array
    pub fn transpose(self) -> Array<ArrayTranspose<ArrayImpl, NDIM>, NDIM> {
        let mut permutation = [0; NDIM];

        for (ind, p) in (0..NDIM).rev().enumerate() {
            permutation[ind] = p;
        }

        self.permute_axes(permutation)
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
