//! Basic Array type
//!
//! [Array] is the basic type for dense calculations in Rlst. The full definition
//! `Array<Item, ArrayImpl, NDIM>` represents a tensor with `NDIM` axes, `Item` as data type
//! (e.g. `f64`), and implemented through `ArrayImpl`.

use crate::dense::base_array::BaseArray;
use crate::dense::data_container::SliceContainer;
use crate::dense::data_container::SliceContainerMut;
use crate::dense::data_container::VectorContainer;
use crate::dense::traits::{
    ChunkedAccess, DefaultIterator, DefaultIteratorMut, NumberOfElements, RandomAccessByRef,
    RandomAccessByValue, RandomAccessMut, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride,
    UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::DataChunk;
use crate::dense::types::RlstScalar;

pub mod empty_axis;
pub mod iterators;
pub mod mult_into;
pub mod operations;
pub mod operators;
pub mod random;
pub mod slice;
pub mod views;

/// A basic dynamically allocated array.
pub type DynamicArray<Item, const NDIM: usize> =
    Array<Item, BaseArray<Item, VectorContainer<Item>, NDIM>, NDIM>;

/// A dynamically allocated array from a data slice.
pub type SliceArray<'a, Item, const NDIM: usize> =
    Array<Item, BaseArray<Item, SliceContainer<'a, Item>, NDIM>, NDIM>;

/// A mutable dynamically allocated array from a data slice.
pub type SliceArrayMut<'a, Item, const NDIM: usize> =
    Array<Item, BaseArray<Item, SliceContainerMut<'a, Item>, NDIM>, NDIM>;

/// A view onto a matrix
pub type ViewArray<'a, Item, ArrayImpl, const NDIM: usize> =
    Array<Item, crate::dense::array::views::ArrayView<'a, Item, ArrayImpl, NDIM>, NDIM>;

/// A mutable view onto a matrix
pub type ViewArrayMut<'a, Item, ArrayImpl, const NDIM: usize> =
    Array<Item, crate::dense::array::views::ArrayViewMut<'a, Item, ArrayImpl, NDIM>, NDIM>;

/// The basic tuple type defining an array.
pub struct Array<Item, ArrayImpl, const NDIM: usize>(ArrayImpl)
where
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>;

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Instantiate a new array from an `ArrayImpl` structure.
    pub fn new(arr: ArrayImpl) -> Self {
        Self(arr)
    }

    /// Return the number of elements in the array.
    pub fn number_of_elements(&self) -> usize {
        self.0.shape().iter().product()
    }
}

impl<Item: RlstScalar, const NDIM: usize> DynamicArray<Item, NDIM> {
    /// Create a new heap allocated array from a given shape.
    pub fn from_shape(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(BaseArray::new(VectorContainer::new(size), shape))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<
        Item: RlstScalar,
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
        Item: RlstScalar,
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
    ) -> Option<crate::dense::types::DataChunk<Self::Item, N>> {
        self.0.get_chunk(chunk_index)
    }
}

impl<
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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
        Item: RlstScalar,
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

/// Create an empty chunk.
pub(crate) fn empty_chunk<const N: usize, Item: RlstScalar>(
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
        Item: RlstScalar,
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
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + RawAccessMut<Item = Item>,
        const NDIM: usize,
    > RawAccessMut for Array<Item, ArrayImpl, NDIM>
{
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > NumberOfElements for Array<Item, ArrayImpl, NDIM>
{
    fn number_of_elements(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + Stride<NDIM>,
        const NDIM: usize,
    > Stride<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ResizeInPlace<NDIM>,
        const NDIM: usize,
    > ResizeInPlace<NDIM> for Array<Item, ArrayImpl, NDIM>
{
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

/// Create an empty array of given type and dimension.
///
/// Empty arrays serve as convenient containers for input into functions that
/// resize an array before filling it with data.
pub fn empty_array<Item: RlstScalar, const NDIM: usize>() -> DynamicArray<Item, NDIM> {
    let shape = [0; NDIM];
    let container = VectorContainer::new(0);
    Array::new(BaseArray::new(container, shape))
}

impl<'a, Item: RlstScalar, const NDIM: usize> SliceArray<'a, Item, NDIM> {
    /// Create a new array from a slice with a given `shape`.
    ///
    /// The `stride` is automatically assumed to be column major.
    pub fn from_shape(slice: &'a [Item], shape: [usize; NDIM]) -> Self {
        Array::new(BaseArray::new(SliceContainer::new(slice), shape))
    }

    /// Create a new array from a slice with a given `shape` and `stride`.
    pub fn from_shape_with_stride(
        slice: &'a [Item],
        shape: [usize; NDIM],
        stride: [usize; NDIM],
    ) -> Self {
        Array::new(BaseArray::new_with_stride(
            SliceContainer::new(slice),
            shape,
            stride,
        ))
    }
}

impl<'a, Item: RlstScalar, const NDIM: usize> SliceArrayMut<'a, Item, NDIM> {
    /// Create a new array from a slice with a given `shape`.
    ///
    /// The `stride` is automatically assumed to be column major.
    pub fn from_shape(slice: &'a mut [Item], shape: [usize; NDIM]) -> Self {
        Array::new(BaseArray::new(SliceContainerMut::new(slice), shape))
    }

    /// Create a new array from a slice with a given `shape` and `stride`.
    pub fn from_shape_with_stride(
        slice: &'a mut [Item],
        shape: [usize; NDIM],
        stride: [usize; NDIM],
    ) -> Self {
        Array::new(BaseArray::new_with_stride(
            SliceContainerMut::new(slice),
            shape,
            stride,
        ))
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > std::fmt::Debug for Array<Item, ArrayImpl, NDIM>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        write!(f, "Array[").unwrap();
        for item in shape {
            write!(f, "{},", item).unwrap();
        }
        write!(f, "]")
    }
}
