//! Coerce an array to a specific dimension if the compiler cannot figure it out
//! directly

use crate::{
    dense::array::Array,
    traits::{
        accessors::{
            RawAccess, RawAccessMut, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
            UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
            UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, Shape, Stride},
    },
    ContainerType,
};

/// A wrapper around an array that coerces its dimension.
pub struct CoerceArray<ArrayImpl, const NDIM: usize, const CDIM: usize> {
    /// The array to coerce.
    arr: Array<ArrayImpl, NDIM>,
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> CoerceArray<ArrayImpl, NDIM, CDIM> {
    /// Create a new `CoerceArray`.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> ContainerType
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> BaseItem
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> Shape<CDIM>
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; CDIM] {
        coerce_index(self.arr.shape())
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> Stride<CDIM>
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: Stride<NDIM>,
{
    #[inline(always)]
    fn stride(&self) -> [usize; CDIM] {
        coerce_index(self.arr.stride())
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> RawAccess
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: RawAccess,
{
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> RawAccessMut
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: RawAccessMut,
{
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandomAccessByValue<CDIM>
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, index: [usize; CDIM]) -> Self::Item {
        self.arr.get_value_unchecked(coerce_index(index))
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandomAccessByRef<CDIM>
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: [usize; CDIM]) -> &Self::Item {
        self.arr.get_unchecked(coerce_index(index))
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandomAccessMut<CDIM>
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, index: [usize; CDIM]) -> &mut Self::Item {
        self.arr.get_unchecked_mut(coerce_index(index))
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandom1DAccessByValue
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.imp().get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandom1DAccessByRef
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByRef,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.imp().get_1d_unchecked(index)
    }
}

impl<ArrayImpl, const NDIM: usize, const CDIM: usize> UnsafeRandom1DAccessMut
    for CoerceArray<ArrayImpl, NDIM, CDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.imp_mut().get_1d_unchecked_mut(index)
    }
}

/// Coerce an index from DIM1 to DIM2 if both dimensions are the same.
#[inline(always)]
fn coerce_index<const DIM1: usize, const DIM2: usize>(index: [usize; DIM1]) -> [usize; DIM2] {
    assert_eq!(DIM1, DIM2);

    // SAFETY: This is safe because we ensure that DIM1 == DIM2 and the types match.

    *unsafe { std::mem::transmute::<&[usize; DIM1], &[usize; DIM2]>(&index) }
}
