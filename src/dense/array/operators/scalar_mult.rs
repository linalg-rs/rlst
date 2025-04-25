//! Container representing multiplication with a scalar

use std::ops::Mul;

use crate::dense::array::{Array, Shape, UnsafeRandomAccessByValue};
use crate::dense::traits::UnsafeRandom1DAccessByValue;
use crate::dense::types::{c32, c64};

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

impl<Scalar: Copy, ArrayImpl, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
    Scalar: Mul<ArrayImpl::Item>,
{
    type Item = Scalar::Output;

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.scalar * self.arr.get_value_unchecked(multi_index)
    }
}

impl<Scalar: Copy, ArrayImpl, const NDIM: usize> UnsafeRandom1DAccessByValue
    for ArrayScalarMult<Scalar, ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue,
    Scalar: Mul<ArrayImpl::Item>,
{
    type Item = Scalar::Output;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.scalar * self.arr.get_value_1d_unchecked(index)
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

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Multiplication by a scalar
    pub fn scalar_mul<Scalar>(
        self,
        other: Scalar,
    ) -> Array<ArrayScalarMult<Scalar, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayScalarMult::new(other, self))
    }
}

macro_rules! impl_scalar_mult {
    ($ScalarType:ty) => {
        impl<ArrayImpl, const NDIM: usize> std::ops::Mul<Array<ArrayImpl, NDIM>> for $ScalarType {
            type Output = Array<ArrayScalarMult<$ScalarType, ArrayImpl, NDIM>, NDIM>;

            fn mul(self, rhs: Array<ArrayImpl, NDIM>) -> Self::Output {
                Array::new(ArrayScalarMult::new(self, rhs))
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);
