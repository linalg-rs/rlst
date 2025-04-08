//! Container representing multiplication with a scalar

use crate::dense::array::{Array, ChunkedAccess, DataChunk, Shape, UnsafeRandomAccessByValue};
use crate::dense::traits::UnsafeRandom1DAccessByValue;
use crate::dense::types::{c32, c64, RlstNum};
use crate::RlstScalar;

/// Scalar multiplication of array
pub struct ArrayScalarMult<
    Item: RlstNum,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    scalar: Item,
    operator: Array<Item, ArrayImpl, NDIM>,
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    /// Create new
    pub fn new(scalar: Item, operator: Array<Item, ArrayImpl, NDIM>) -> Self {
        Self { scalar, operator }
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.scalar * self.operator.get_value_unchecked(multi_index)
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        if let Some(mut chunk) = self.operator.get_chunk(chunk_index) {
            for item in &mut chunk.data {
                *item *= self.scalar;
            }
            Some(chunk)
        } else {
            None
        }
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Multiplication by a scalar
    pub fn scalar_mul(
        self,
        other: Item,
    ) -> Array<Item, ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayScalarMult::new(other, self))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > std::ops::Neg for Array<Item, ArrayImpl, NDIM>
{
    type Output = Array<Item, ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM>;

    fn neg(self) -> Self::Output {
        let minus_one = -<Item as num::One>::one();
        Array::new(ArrayScalarMult::new(minus_one, self))
    }
}

impl<
        Item: RlstNum,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandom1DAccessByValue<Item = Item>,
        const NDIM: usize,
    > UnsafeRandom1DAccessByValue for ArrayScalarMult<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.scalar * self.operator.get_value_1d_unchecked(index)
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
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);
