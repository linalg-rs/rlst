//! Reference to an array.
//!
//! A reference is an owned struct that holds a reference to an array. It is used to
//! pass arrays to functions without transferring ownership.

use crate::{
    dense::array::Array,
    traits::{
        accessors::{
            RawAccess, RawAccessMut, UnsafeRandom1DAccessByRef, UnsafeRandom1DAccessByValue,
            UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
            UnsafeRandomAccessMut,
        },
        base_operations::{BaseItem, ResizeInPlace, Shape, Stride},
    },
    AsOwnedRefType, AsOwnedRefTypeMut, ContainerType,
};

/// Basic structure for a `View`
pub struct ArrayRef<'a, ArrayImpl, const NDIM: usize>(&'a Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> ContainerType for ArrayRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<'a, ArrayImpl, const NDIM: usize> BaseItem for ArrayRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayRef<'a, ArrayImpl, NDIM> {
    /// Create new view
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayRef<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayRef<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

/////////// ArrayRefMut

/// Mutable array view
pub struct ArrayRefMut<'a, ArrayImpl, const NDIM: usize>(&'a mut Array<ArrayImpl, NDIM>);

impl<'a, ArrayImpl, const NDIM: usize> BaseItem for ArrayRefMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayRefMut<'a, ArrayImpl, NDIM> {
    /// Create new mutable view
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for ArrayRefMut<'_, ArrayImpl, NDIM> {
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl: ResizeInPlace<NDIM>, const NDIM: usize> ResizeInPlace<NDIM>
    for ArrayRefMut<'_, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

impl<ArrayImpl, const NDIM: usize> AsOwnedRefType for Array<ArrayImpl, NDIM> {
    type RefType<'a>
        = Array<ArrayRef<'a, ArrayImpl, NDIM>, NDIM>
    where
        Self: 'a;

    fn r(&self) -> Self::RefType<'_> {
        Array::new(ArrayRef::new(self))
    }
}

impl<ArrayImpl, const NDIM: usize> AsOwnedRefTypeMut for Array<ArrayImpl, NDIM> {
    type RefTypeMut<'a>
        = Array<ArrayRefMut<'a, ArrayImpl, NDIM>, NDIM>
    where
        Self: 'a;

    fn r_mut(&mut self) -> Self::RefTypeMut<'_> {
        Array::new(ArrayRefMut::new(self))
    }
}
