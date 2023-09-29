//! Container representing multiplication with a scalar

use rlst_common::types::{c32, c64};
use std::marker::PhantomData;

use crate::array::*;

pub struct ArrayToComplex<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as Scalar>::Real> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<<Item as Scalar>::Real, ArrayImpl, NDIM>,
    _marker: PhantomData<Item>,
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as Scalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayToComplex<Item, ArrayImpl, NDIM>
{
    pub fn new(operator: Array<<Item as Scalar>::Real, ArrayImpl, NDIM>) -> Self {
        Self {
            operator,
            _marker: PhantomData,
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as Scalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayToComplex<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        <Item as Scalar>::from_real(self.operator.get_value_unchecked(multi_index))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as Scalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayToComplex<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as Scalar>::Real>
            + Shape<NDIM>
            + ChunkedAccess<N, Item = <Item as Scalar>::Real>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for ArrayToComplex<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>> {
        if let Some(chunk) = self.operator.get_chunk(chunk_index) {
            let mut data = [<Item as num::Zero>::zero(); N];

            for (d, &c) in data.iter_mut().zip(chunk.data.iter()) {
                *d = <Item as Scalar>::from_real(c);
            }
            Some(DataChunk::<Item, N> {
                data,
                start_index: chunk.start_index,
                valid_entries: chunk.valid_entries,
            })
        } else {
            None
        }
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f32> + Shape<NDIM>, const NDIM: usize>
    rlst_common::traits::ToComplex for Array<f32, ArrayImpl, NDIM>
{
    type Out = Array<c32, ArrayToComplex<c32, ArrayImpl, NDIM>, NDIM>;

    fn to_complex(self) -> Self::Out {
        Array::new(ArrayToComplex::new(self))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f64> + Shape<NDIM>, const NDIM: usize>
    rlst_common::traits::ToComplex for Array<f64, ArrayImpl, NDIM>
{
    type Out = Array<c64, ArrayToComplex<c64, ArrayImpl, NDIM>, NDIM>;

    fn to_complex(self) -> Self::Out {
        Array::new(ArrayToComplex::new(self))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = c64> + Shape<NDIM>, const NDIM: usize>
    rlst_common::traits::ToComplex for Array<c64, ArrayImpl, NDIM>
{
    type Out = Self;

    fn to_complex(self) -> Self::Out {
        self
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = c32> + Shape<NDIM>, const NDIM: usize>
    rlst_common::traits::ToComplex for Array<c32, ArrayImpl, NDIM>
{
    type Out = Self;

    fn to_complex(self) -> Self::Out {
        self
    }
}
