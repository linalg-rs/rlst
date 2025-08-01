//! Container representing multiplication with a scalar

use std::ops::Neg;

use crate::ContainerTypeHint;
use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, base_operations::BaseItem},
};

/// Scalar multiplication of array
pub struct ArrayNeg<ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
}

impl<ArrayImpl, const NDIM: usize> ArrayNeg<ArrayImpl, NDIM> {
    /// Create new
    pub fn new(arr: Array<ArrayImpl, NDIM>) -> Self {
        Self { arr }
    }
}

impl<ArrayImpl, const NDIM: usize> ContainerTypeHint for ArrayNeg<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerTypeHint,
{
    type TypeHint = ArrayImpl::TypeHint;
}

impl<ArrayImpl, const NDIM: usize> BaseItem for ArrayNeg<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM> for ArrayNeg<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + BaseItem,
    ArrayImpl::Item: Neg<Output = ArrayImpl::Item>,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.arr.get_value_unchecked(multi_index).neg()
    }
}

impl<ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue for ArrayNeg<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + BaseItem,
    ArrayImpl::Item: Neg<Output = ArrayImpl::Item>,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.arr.get_value_1d_unchecked(index).neg()
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for ArrayNeg<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}
