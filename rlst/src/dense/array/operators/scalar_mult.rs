//! Container representing multiplication with a scalar

use std::ops::Mul;

use crate::base_types::{c32, c64};

use crate::{
    dense::array::{Array, Shape, UnsafeRandomAccessByValue},
    traits::{accessors::UnsafeRandom1DAccessByValue, base_operations::BaseItem},
};
use crate::{ContainerType, ScalarMul};

/// Scalar multiplication of array
pub struct ArrayScalarMult<Scalar, ArrayImpl, const NDIM: usize> {
    arr: Array<ArrayImpl, NDIM>,
    scalar: Scalar,
}

impl<Scalar, ArrayImpl, const NDIM: usize> ArrayScalarMult<Scalar, ArrayImpl, NDIM> {
    /// Create new
    pub fn new(scalar: Scalar, arr: Array<ArrayImpl, NDIM>) -> Self {
        Self { arr, scalar }
    }
}

impl<Scalar, ArrayImpl, const NDIM: usize> ContainerType
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType,
{
    type Type = ArrayImpl::Type;
}

impl<Scalar, ArrayImpl, const NDIM: usize> BaseItem for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Scalar>,
    Scalar: Copy + Default,
{
    type Item = Scalar;
}

impl<Scalar, ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + BaseItem<Item = Scalar>,
    Scalar: Mul<ArrayImpl::Item, Output = Scalar>,
    Scalar: Copy + Default,
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.scalar * self.arr.get_value_unchecked(multi_index)
    }
}

impl<Scalar, ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + BaseItem<Item = Scalar>,
    Scalar: Mul<ArrayImpl::Item, Output = Scalar>,
    Scalar: Copy + Default,
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.scalar * self.arr.imp().get_value_1d_unchecked(index)
    }
}

impl<Scalar, ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM>
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
{
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.arr.shape()
    }
}
