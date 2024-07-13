//! Container representing multiplication with a scalar

use crate::dense::types::{c32, c64};
use std::marker::PhantomData;

use crate::dense::array::{Array, ChunkedAccess, DataChunk, Shape, UnsafeRandomAccessByValue};

use crate::dense::types::RlstScalar;

/// Array to complex
pub struct ArrayToComplex<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as RlstScalar>::Real> + Shape<NDIM>,
    const NDIM: usize,
> {
    operator: Array<<Item as RlstScalar>::Real, ArrayImpl, NDIM>,
    _marker: PhantomData<Item>,
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as RlstScalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayToComplex<Item, ArrayImpl, NDIM>
{
    /// Create new
    pub fn new(operator: Array<<Item as RlstScalar>::Real, ArrayImpl, NDIM>) -> Self {
        Self {
            operator,
            _marker: PhantomData,
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as RlstScalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayToComplex<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        <Item as RlstScalar>::from_real(self.operator.get_value_unchecked(multi_index))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as RlstScalar>::Real> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayToComplex<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.operator.shape()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = <Item as RlstScalar>::Real>
            + Shape<NDIM>
            + ChunkedAccess<N, Item = <Item as RlstScalar>::Real>,
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
                *d = <Item as RlstScalar>::from_real(c);
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
    Array<f32, ArrayImpl, NDIM>
{
    /// Convert to complex
    pub fn to_complex(self) -> Array<c32, ArrayToComplex<c32, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayToComplex::new(self))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = f64> + Shape<NDIM>, const NDIM: usize>
    Array<f64, ArrayImpl, NDIM>
{
    /// Convert to complex
    pub fn to_complex(self) -> Array<c64, ArrayToComplex<c64, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayToComplex::new(self))
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = c64> + Shape<NDIM>, const NDIM: usize>
    Array<c64, ArrayImpl, NDIM>
{
    /// Convert to complex
    pub fn to_complex(self) -> Self {
        self
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = c32> + Shape<NDIM>, const NDIM: usize>
    Array<c32, ArrayImpl, NDIM>
{
    /// Convert to complex
    pub fn to_complex(self) -> Self {
        self
    }
}
