//! Basic Array type

use crate::base_array::BaseArray;
use crate::data_container::VectorContainer;
use rlst_common::traits::ChunkedAccess;
use rlst_common::traits::NumberOfElements;
use rlst_common::traits::RandomAccessByRef;
use rlst_common::traits::RandomAccessMut;
use rlst_common::traits::RawAccess;
use rlst_common::traits::RawAccessMut;
use rlst_common::traits::Shape;
use rlst_common::traits::Stride;
use rlst_common::traits::UnsafeRandomAccessByRef;
use rlst_common::traits::UnsafeRandomAccessByValue;
use rlst_common::traits::UnsafeRandomAccessMut;
use rlst_common::types::DataChunk;
use rlst_common::types::Scalar;

pub mod dot;
pub mod iterators;
pub mod operations;
pub mod operators;
pub mod random;
pub mod slice;
pub mod views;

pub type DynamicArray<Item, const NDIM: usize> =
    Array<Item, BaseArray<Item, VectorContainer<Item>, NDIM>, NDIM>;

/// The basic tuple type defining an array.
pub struct Array<Item, ArrayImpl, const NDIM: usize>(ArrayImpl)
where
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>;

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn new(arr: ArrayImpl) -> Self {
        Self(arr)
    }

    pub fn new_dynamic_like_self(&self) -> DynamicArray<Item, NDIM> {
        DynamicArray::<Item, NDIM>::from_shape(self.shape())
    }

    pub fn number_of_elements(&self) -> usize {
        self.0.shape().iter().product()
    }
}

impl<Item: Scalar, const NDIM: usize> DynamicArray<Item, NDIM> {
    pub fn from_shape(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(BaseArray::new(VectorContainer::new(size), shape))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const NDIM: usize,
        const N: usize,
    > ChunkedAccess<N> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    fn get_chunk(
        &self,
        chunk_index: usize,
    ) -> Option<rlst_common::types::DataChunk<Self::Item, N>> {
        self.0.get_chunk(chunk_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(multi_index)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > std::ops::Index<[usize; NDIM]> for Array<Item, ArrayImpl, NDIM>
{
    type Output = Item;
    #[inline]
    fn index(&self, index: [usize; NDIM]) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > std::ops::IndexMut<[usize; NDIM]> for Array<Item, ArrayImpl, NDIM>
{
    #[inline]
    fn index_mut(&mut self, index: [usize; NDIM]) -> &mut Self::Output {
        self.0.get_mut(index).unwrap()
    }
}

pub(crate) fn empty_chunk<const N: usize, Item: Scalar>(
    chunk_index: usize,
    nelements: usize,
) -> Option<DataChunk<Item, N>> {
    let start_index = N * chunk_index;
    if start_index >= nelements {
        return None;
    }
    let end_index = (1 + chunk_index) * N;
    let valid_entries = if end_index > nelements {
        nelements - start_index
    } else {
        N
    };
    Some(DataChunk {
        data: [<Item as num::Zero>::zero(); N],
        start_index,
        valid_entries,
    })
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + RawAccess<Item = Item>,
        const NDIM: usize,
    > RawAccess for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;

    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > RawAccessMut for Array<Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > NumberOfElements for Array<Item, ArrayImpl, NDIM>
{
    fn number_of_elements(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + Stride<NDIM>,
        const NDIM: usize,
    > Stride<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}
