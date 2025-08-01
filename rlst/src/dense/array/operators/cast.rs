//! Cast an array to another type

use std::marker::PhantomData;

use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, base_operations::BaseItem},
    ContainerType,
};

/// Array to complex
pub struct ArrayCast<Target, ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    _marker: PhantomData<Target>,
}

impl<Target, ArrayImpl, const NDIM: usize> ArrayCast<Target, ArrayImpl, NDIM> {
    /// Create new
    pub fn new(operator: Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr: operator,
            _marker: PhantomData,
        }
    }
}

impl<Target, ArrayImpl, const NDIM: usize> ContainerType for ArrayCast<Target, ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<Target, ArrayImpl, const NDIM: usize> BaseItem for ArrayCast<Target, ArrayImpl, NDIM> {
    type Item = Target;
}

impl<Target, ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize>
    UnsafeRandomAccessByValue<NDIM> for ArrayCast<Target, ArrayImpl, NDIM>
where
    ArrayImpl::Item: num::NumCast,
    Target: num::NumCast,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        num::cast::<ArrayImpl::Item, Target>(self.arr.get_value_unchecked(multi_index)).unwrap()
    }
}

impl<Target, ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM>
    for ArrayCast<Target, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Cast array to type `T`.
    pub fn cast<Target>(self) -> Array<ArrayCast<Target, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayCast::new(self))
    }
}

impl<Target, ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayCast<Target, ArrayImpl, NDIM>
where
    ArrayImpl::Item: num::NumCast,
    Target: num::NumCast,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        num::cast::<ArrayImpl::Item, Target>(self.arr.get_value_1d_unchecked(index)).unwrap()
    }
}
