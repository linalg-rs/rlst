//! Container representing multiplication with a scalar

use crate::array::*;
use rlst_common::types::*;

pub struct ArrayScalarMult<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    scalar: Item,
    operator: Array<Item, ArrayImpl, NDIM>,
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    pub fn new(scalar: Item, operator: Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { scalar, operator }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.scalar * self.operator.get_value_unchecked(indices)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

macro_rules! impl_scalar_mult {
    ($ScalarType:ty) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = $ScalarType> + Shape<NDIM>,
                const NDIM: usize,
            > std::ops::Mul<Array<$ScalarType, ArrayImpl, NDIM>> for $ScalarType
        {
            type Output = Array<$ScalarType, ArrayScalarMult<$ScalarType, ArrayImpl, NDIM>, NDIM>;

            fn mul(self, rhs: Array<$ScalarType, ArrayImpl, NDIM>) -> Self::Output {
                Array::new(ArrayScalarMult::new(self, rhs))
            }
        }

        impl<
                ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = $ScalarType> + Shape<NDIM>,
                const NDIM: usize,
            > std::ops::Mul<$ScalarType> for Array<$ScalarType, ArrayImpl, NDIM>
        {
            type Output = Array<$ScalarType, ArrayScalarMult<$ScalarType, ArrayImpl, NDIM>, NDIM>;

            fn mul(self, rhs: $ScalarType) -> Self::Output {
                rhs * self
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
