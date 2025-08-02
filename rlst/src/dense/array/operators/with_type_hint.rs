//! Change the type hint of an array to a different type.

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
    ContainerType, ContainerTypeRepr,
};

/// A wrapper around an array that coerces its dimension.
pub struct WithTypeHint<ArrayImpl, TypeHint, const NDIM: usize> {
    /// The array whose type hint is to be changed.
    arr: Array<ArrayImpl, NDIM>,
    _type_hint: std::marker::PhantomData<TypeHint>,
}

impl<ArrayImpl, TypeHint, const NDIM: usize> WithTypeHint<ArrayImpl, TypeHint, NDIM> {
    /// Create a new `WithTypeHint`.
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            _type_hint: std::marker::PhantomData,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Coerce the array to a specific item type and dimension.
    pub fn with_container_type<TypeHint: ContainerTypeRepr>(
        self,
    ) -> Array<WithTypeHint<ArrayImpl, TypeHint, NDIM>, NDIM> {
        Array::new(WithTypeHint::new(self))
    }
}

impl<ArrayImpl, TypeHint: ContainerTypeRepr, const NDIM: usize> ContainerType
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = TypeHint;
}

impl<ArrayImpl, TypeHint, const NDIM: usize> BaseItem for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, TypeHint, const NDIM: usize> Shape<NDIM> for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> Stride<NDIM>
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: Stride<NDIM>,
{
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.arr.stride()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> RawAccess for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: RawAccess,
{
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.arr.data()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> RawAccessMut
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: RawAccessMut,
{
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.arr.data_mut()
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(multi_index)
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.arr.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessByValue
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessByRef
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByRef,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.arr.get_1d_unchecked(index)
    }
}

impl<ArrayImpl, TypeHint, const NDIM: usize> UnsafeRandom1DAccessMut
    for WithTypeHint<ArrayImpl, TypeHint, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut,
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.arr.get_1d_unchecked_mut(index)
    }
}
