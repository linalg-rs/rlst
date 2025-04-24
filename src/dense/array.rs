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
    NumberOfElements, RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, ResizeInPlace,
    Shape, Stride, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

use super::strided_base_array::StridedBaseArray;
use super::traits::ArrayIterator;
use super::traits::ArrayIteratorMut;
use super::traits::UnsafeRandom1DAccessByRef;
use super::traits::UnsafeRandom1DAccessByValue;
use super::traits::UnsafeRandom1DAccessMut;

// pub mod empty_axis;
// pub mod iterators;
// pub mod mult_into;
// pub mod operations;
// pub mod operators;
// pub mod random;
// pub mod rank1_array;
pub mod reference;
// pub mod slice;
// pub mod views;

/// A basic dynamically allocated array.
pub type DynamicArray<Item, const NDIM: usize> =
    Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM>;

/// A dynamically allocated array from a data slice.
pub type SliceArray<'a, Item, const NDIM: usize> =
    Array<BaseArray<SliceContainer<'a, Item>, NDIM>, NDIM>;

/// A dynamically allocated array from a data slice with a given stride.
pub type StridedSliceArray<'a, Item, const NDIM: usize> =
    Array<StridedBaseArray<SliceContainer<'a, Item>, NDIM>, NDIM>;

/// A mutable dynamically allocated array from a data slice.
pub type SliceArrayMut<'a, Item, const NDIM: usize> =
    Array<BaseArray<SliceContainerMut<'a, Item>, NDIM>, NDIM>;

/// A mutable dynamically allocated array from a data slice with a given stride.
pub type StridedSliceArrayMut<'a, Item, const NDIM: usize> =
    Array<StridedBaseArray<SliceContainerMut<'a, Item>, NDIM>, NDIM>;

// /// A view onto a matrix
// pub type ViewArray<'a, Item, ArrayImpl, const NDIM: usize> =
//     Array<crate::dense::array::views::ArrayView<'a, Item, ArrayImpl, NDIM>, NDIM>;

// /// A mutable view onto a matrix
// pub type ViewArrayMut<'a, Item, ArrayImpl, const NDIM: usize> =
//     Array<Item, crate::dense::array::views::ArrayViewMut<'a, Item, ArrayImpl, NDIM>, NDIM>;

/// The basic tuple type defining an array.
pub struct Array<ArrayImpl, const NDIM: usize>(ArrayImpl);

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Instantiate a new array from an `ArrayImpl` structure.
    pub fn new(arr: ArrayImpl) -> Self {
        Self(arr)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Return the number of elements in the array.
    #[inline(always)]
    pub fn number_of_elements(&self) -> usize {
        self.0.shape().iter().product()
    }
}

impl<Item: Clone + Default, const NDIM: usize> Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
    /// Create a new heap allocated array from a given shape.
    #[inline(always)]
    pub fn from_shape(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(BaseArray::new(VectorContainer::new(size), shape))
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for Array<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item {
        self.0.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl: RandomAccessByRef<NDIM>, const NDIM: usize> std::ops::Index<[usize; NDIM]>
    for Array<ArrayImpl, NDIM>
{
    type Output = ArrayImpl::Item;
    #[inline(always)]
    fn index(&self, index: [usize; NDIM]) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl<ArrayImpl: RandomAccessMut<NDIM>, const NDIM: usize> std::ops::IndexMut<[usize; NDIM]>
    for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: std::ops::Index<[usize; NDIM], Output = ArrayImpl::Item>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; NDIM]) -> &mut ArrayImpl::Item {
        self.0.get_mut(index).unwrap()
    }
}

impl<ArrayImpl: RawAccess, const NDIM: usize> RawAccess for Array<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for Array<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.0.data_mut()
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> NumberOfElements for Array<ArrayImpl, NDIM> {
    #[inline(always)]
    fn number_of_elements(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<ArrayImpl: Stride<NDIM>, const NDIM: usize> Stride<NDIM> for Array<ArrayImpl, NDIM> {
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl: ResizeInPlace<NDIM>, const NDIM: usize> ResizeInPlace<NDIM>
    for Array<ArrayImpl, NDIM>
{
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

/// Create an empty array of given type and dimension.
///
/// Empty arrays serve as convenient containers for input into functions that
/// resize an array before filling it with data.
pub fn empty_array<Item: Clone + Default, const NDIM: usize>(
) -> Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
    let shape = [0; NDIM];
    let container = VectorContainer::new(0);
    Array::new(BaseArray::new(container, shape))
}

impl<'a, Item, const NDIM: usize> StridedSliceArray<'a, Item, NDIM> {
    /// Create a new array from a slice with a given `shape` and `stride`.
    ///
    pub fn from_shape_and_stride(
        slice: &'a [Item],
        shape: [usize; NDIM],
        stride: [usize; NDIM],
    ) -> Self {
        Array::new(StridedBaseArray::new(
            SliceContainer::new(slice),
            shape,
            stride,
        ))
    }
}

impl<'a, Item, const NDIM: usize> StridedSliceArrayMut<'a, Item, NDIM> {
    /// Create a new array from a slice with a given `shape` and `stride`.
    ///
    /// The `stride` is automatically assumed to be column major.
    pub fn from_shape_and_stride(
        slice: &'a mut [Item],
        shape: [usize; NDIM],
        stride: [usize; NDIM],
    ) -> Self {
        Array::new(StridedBaseArray::new(
            SliceContainerMut::new(slice),
            shape,
            stride,
        ))
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> std::fmt::Debug for Array<ArrayImpl, NDIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.shape();
        write!(f, "Array[").unwrap();
        for item in shape {
            write!(f, "{},", item).unwrap();
        }
        write!(f, "]")
    }
}

impl<ArrayImpl: ArrayIterator, const NDIM: usize> ArrayIterator for Array<ArrayImpl, NDIM> {
    type Item = ArrayImpl::Item;

    type Iter<'a>
        = <ArrayImpl as ArrayIterator>::Iter<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn iter(&self) -> Self::Iter<'_> {
        self.0.iter()
    }
}

impl<ArrayImpl, const NDIM: usize> ArrayIteratorMut for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ArrayIteratorMut,
{
    type Item = ArrayImpl::Item;

    type IterMut<'a>
        = <ArrayImpl as ArrayIteratorMut>::IterMut<'a>
    where
        Self: 'a;

    #[inline(always)]
    fn iter_mut(&self) -> Self::IterMut<'_> {
        self.0.iter_mut()
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for Array<ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(index)
    }
}
