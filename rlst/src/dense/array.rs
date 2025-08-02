//! Basic Array type
//!
//! [Array] is the basic type for dense calculations in Rlst. The full definition
//! `Array<ArrayImpl, NDIM>` represents a tensor with `NDIM` axes implemented through
//! the implementeation type `ArrayImpl`.

use std::ops::AddAssign;

use iterators::{
    ArrayDefaultIteratorByRef, ArrayDefaultIteratorByValue, ArrayDefaultIteratorMut,
    ArrayDiagIteratorByRef, ArrayDiagIteratorByValue, ArrayDiagIteratorMut,
};
use itertools::izip;
use operators::unary_op::ArrayUnaryOperator;

use crate::{
    dense::{
        base_array::BaseArray,
        data_container::{SliceContainer, SliceContainerMut, VectorContainer},
        strided_base_array::StridedBaseArray,
    },
    traits::{
        accessors::{
            RandomAccessByRef, RandomAccessMut, RawAccess, RawAccessMut, UnsafeRandom1DAccessByRef,
            UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef,
            UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
        },
        base_operations::{
            BaseItem, FillFromResize, Len, NumberOfElements, ResizeInPlace, Shape, Stride,
        },
        data_container::ContainerType,
    },
    Abs, AbsSquare, AsOwnedRefType, AsOwnedRefTypeMut, Conj, DispatchEval, DispatchEvalRowMajor,
    EvaluateObject, EvaluateRowMajorArray, FillFrom, FillFromIter, Max, RandomAccessByValue, Sqrt,
    Stack, SumFrom, Unknown,
};

use super::{data_container::ArrayContainer, layout::row_major_stride_from_shape};

pub mod empty_axis;
pub mod iterators;
pub mod mult_into;
pub mod operations;
pub mod operators;
pub mod random;
// pub mod rank1_array;
pub mod flattened;
pub mod reference;
pub mod row_major_view;
pub mod slice;
pub mod subview;

/// A basic dynamically allocated array.
pub type DynArray<Item, const NDIM: usize> = Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM>;

/// A basic dynamically allocated array with a given stride.
pub type StridedDynArray<Item, const NDIM: usize> =
    Array<StridedBaseArray<VectorContainer<Item>, NDIM>, NDIM>;

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

/// The reference type associated with an array.
pub type RefType<'a, Arr> = <Arr as AsOwnedRefType>::RefType<'a>;

/// The mutable reference type associated with an array.
pub type RefTypeMut<'a, Arr> = <Arr as AsOwnedRefTypeMut>::RefTypeMut<'a>;

/// The basic tuple type defining an array.
pub struct Array<ArrayImpl, const NDIM: usize>(ArrayImpl);

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Instantiate a new array from an `ArrayImpl` structure.
    pub fn new(arr: ArrayImpl) -> Self {
        assert!(
            NDIM > 0,
            "Array dimension must be greater than 0, got {NDIM}"
        );
        Self(arr)
    }

    /// Return a reference to the implementation.
    #[inline(always)]
    pub fn inner(&self) -> &ArrayImpl {
        &self.0
    }

    /// Return a mutable reference to the implementation.
    #[inline(always)]
    pub fn inner_mut(&mut self) -> &mut ArrayImpl {
        &mut self.0
    }

    /// Extract the inner implementation.
    #[inline(always)]
    pub fn into_inner(self) -> ArrayImpl {
        self.0
    }
}

impl<ArrayImpl, const NDIM: usize> BaseItem for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Return the number of elements in the array.
    #[inline(always)]
    pub fn number_of_elements(&self) -> usize {
        self.0.shape().iter().product()
    }
}

impl<Item: Copy + Default, const NDIM: usize> Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
    /// Create a new heap allocated array from a given `shape`.
    #[inline(always)]
    pub fn from_shape(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(BaseArray::new(VectorContainer::new(size), shape))
    }

    /// Create a new heap allocated array by providing a `shape` and a vector of `data`.
    ///
    /// The number of elements in the vector must be compatible with the given shape.
    /// Otherwise, an assertion error is triggered.
    #[inline(always)]
    pub fn from_shape_and_vec(shape: [usize; NDIM], data: Vec<Item>) -> Self {
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length does not match the shape: {} != {}",
            data.len(),
            shape.iter().product::<usize>()
        );
        Self::new(BaseArray::new(VectorContainer::from_vec(data), shape))
    }
}

impl<Item: Copy + Default> From<&[Item]> for DynArray<Item, 1> {
    fn from(value: &[Item]) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value.to_vec())
    }
}

impl<Item: Copy + Default> From<Vec<Item>> for DynArray<Item, 1> {
    fn from(value: Vec<Item>) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value)
    }
}

impl<Item: Copy + Default, const NDIM: usize>
    Array<StridedBaseArray<VectorContainer<Item>, NDIM>, NDIM>
{
    /// Create a new heap allocated array from a given `shape` and `stride`.
    #[inline(always)]
    pub fn from_shape_with_stride(shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(StridedBaseArray::new(
            VectorContainer::new(size),
            shape,
            stride,
        ))
    }
}

impl<Item: Copy + Default, const NDIM: usize>
    Array<StridedBaseArray<VectorContainer<Item>, NDIM>, NDIM>
{
    /// Create a new heap allocated row-major array.
    #[inline(always)]
    pub fn row_major(shape: [usize; NDIM]) -> Self {
        let size = shape.iter().product();
        Self::new(StridedBaseArray::new(
            VectorContainer::new(size),
            shape,
            row_major_stride_from_shape(shape),
        ))
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Return the shape of an array.
    ///
    /// # Traits:
    /// - [Shape](crate::Shape)
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Return the length of an array.
    ///
    /// For more than one dimension the length is the number of elements.
    /// # Traits:
    /// - [Shape](crate::Shape)
    #[inline(always)]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM>,
{
    /// Get an element by value.
    ///
    /// # Safety
    /// `multi_index` must be in range.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByValue](crate::UnsafeRandomAccessByValue)
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> ArrayImpl::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM>,
{
    /// Get an element by reference.
    ///
    /// # Safety
    /// `multi_index` must be in range.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByRef](crate::UnsafeRandomAccessByRef)
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &ArrayImpl::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM>,
{
    /// Get an element by mutable reference.
    ///
    /// # Safety
    /// `multi_index` must be in range.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessMut](crate::UnsafeRandomAccessMut)
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut ArrayImpl::Item {
        self.0.get_unchecked_mut(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RandomAccessByValue<NDIM>,
{
    /// Get an element by value.
    ///
    /// # Traits
    /// - [RandomAccessByValue](crate::RandomAccessByValue)
    #[inline(always)]
    unsafe fn get_value(&self, multi_index: [usize; NDIM]) -> Option<ArrayImpl::Item> {
        self.0.get_value(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RandomAccessByRef<NDIM>,
{
    /// Get an element by reference.
    ///
    /// # Traits
    /// - [RandomAccessByRef](crate::RandomAccessByRef)
    #[inline(always)]
    unsafe fn get(&self, multi_index: [usize; NDIM]) -> Option<&ArrayImpl::Item> {
        self.0.get(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RandomAccessMut<NDIM>,
{
    /// Get an element by mutable reference.
    ///
    /// # Traits
    /// - [RandomAccessMut](crate::RandomAccessMut)
    #[inline(always)]
    unsafe fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut ArrayImpl::Item> {
        self.0.get_mut(multi_index)
    }
}

impl<ArrayImpl, const NDIM: usize> std::ops::Index<[usize; NDIM]> for Array<ArrayImpl, NDIM>
where
    ArrayImpl: RandomAccessByRef<NDIM>,
{
    type Output = ArrayImpl::Item;
    #[inline(always)]
    fn index(&self, index: [usize; NDIM]) -> &Self::Output {
        self.0.get(index).unwrap()
    }
}

impl<ArrayImpl, const NDIM: usize> std::ops::IndexMut<[usize; NDIM]> for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: std::ops::Index<[usize; NDIM], Output = ArrayImpl::Item>,
    ArrayImpl: RandomAccessMut<NDIM>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; NDIM]) -> &mut ArrayImpl::Item {
        self.0.get_mut(index).unwrap()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RawAccess,
{
    /// Return the raw data as slice.
    ///
    /// # Traits
    /// - [RawAccess](crate::RawAccess)
    #[inline(always)]
    fn data(&self) -> &[ArrayImpl::Item] {
        self.0.data()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RawAccessMut,
{
    /// Return the raw data as slice.
    ///
    /// # Traits
    /// - [RawAccessMut](crate::RawAccessMut)
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [ArrayImpl::Item] {
        self.0.data_mut()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Stride<NDIM>,
{
    /// Return the stride.
    ///
    /// # Traits
    /// - [Stride](crate::Stride)
    #[inline(always)]
    fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: ResizeInPlace<NDIM>,
{
    /// Resize the array to the new `shape`.
    ///
    /// The content of the array will be lost upon resizing.
    ///
    /// # Traits
    /// - [ResizeInPlace](crate::ResizeInPlace)
    fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

/// Create an empty array of given type and dimension.
///
/// Empty arrays serve as convenient containers for input into functions that
/// resize an array before filling it with data.
pub fn empty_array<Item: Copy + Default, const NDIM: usize>(
) -> Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM> {
    let shape = [0; NDIM];
    let container = VectorContainer::new(0);
    Array::new(BaseArray::new(container, shape))
}

impl<'a, Item, const NDIM: usize> SliceArray<'a, Item, NDIM> {
    /// Create a new array from `slice` with a given `shape`.
    pub fn from_shape(slice: &'a [Item], shape: [usize; NDIM]) -> Self {
        assert_eq!(
            slice.len(),
            shape.iter().product::<usize>(),
            "Slice length does not match the shape: {} != {}",
            slice.len(),
            shape.iter().product::<usize>()
        );
        Array::new(BaseArray::new(SliceContainer::new(slice), shape))
    }
}

impl<'a, Item, const NDIM: usize> SliceArrayMut<'a, Item, NDIM> {
    /// Create a new array from mutable `slice` with a given `shape`.
    pub fn from_shape(slice: &'a mut [Item], shape: [usize; NDIM]) -> Self {
        assert_eq!(
            slice.len(),
            shape.iter().product::<usize>(),
            "Slice length does not match the shape: {} != {}",
            slice.len(),
            shape.iter().product::<usize>()
        );
        Array::new(BaseArray::new(SliceContainerMut::new(slice), shape))
    }
}

impl<'a, Item, const NDIM: usize> StridedSliceArray<'a, Item, NDIM> {
    /// Create a new array from `slice` with a given `shape` and `stride`.
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
    /// Create a new array from `slice` with a given `shape` and `stride`.
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
            write!(f, "{item},").unwrap();
        }
        write!(f, "]")
    }
}

impl<Item, const NDIM: usize> DynArray<Item, NDIM>
where
    Item: Copy + Default,
{
    /// Create a new array and fill with values from `arr`.
    pub fn new_from<ArrayImpl>(arr: &Array<ArrayImpl, NDIM>) -> Self
    where
        DynArray<Item, NDIM>: FillFromResize<Array<ArrayImpl, NDIM>>,
    {
        let mut output = empty_array::<Item, NDIM>();
        output.fill_from_resize(arr);
        output
    }
}

/// A dispatcher for evaluating arrays.
///
/// This dispatcher enables evaluating arrays either into a new heap
/// allocated array or a new stack allocated array depending on the type hint
/// of the array.
pub struct EvalDispatcher<TypeHint, ArrayImpl> {
    _type_hint: std::marker::PhantomData<(TypeHint, ArrayImpl)>,
}

impl<TypeHint, ArrayImpl> Default for EvalDispatcher<TypeHint, ArrayImpl> {
    fn default() -> Self {
        Self {
            _type_hint: std::marker::PhantomData,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> DispatchEval<NDIM> for EvalDispatcher<Unknown, ArrayImpl>
where
    ArrayImpl::Item: Copy + Default,
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Output = DynArray<ArrayImpl::Item, NDIM>;

    type ArrayImpl = ArrayImpl;

    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output {
        let mut output = DynArray::new_from(arr);
        output.fill_from(arr);
        output
    }
}

impl<ArrayImpl, const NDIM: usize, const N: usize> DispatchEval<NDIM>
    for EvalDispatcher<Stack<N>, ArrayImpl>
where
    ArrayImpl::Item: Copy + Default,
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Output = Array<BaseArray<ArrayContainer<ArrayImpl::Item, N>, NDIM>, NDIM>;

    type ArrayImpl = ArrayImpl;

    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output {
        let mut output = Array::new(BaseArray::new(
            ArrayContainer::<ArrayImpl::Item, N>::new(),
            arr.shape(),
        ));
        output.fill_from(arr);
        output
    }
}

/// A dispatcher for evaluating arrays into row-major order.
pub struct EvalRowMajorDispatcher<TypeHint, ArrayImpl> {
    _type_hint: std::marker::PhantomData<(TypeHint, ArrayImpl)>,
}

impl<TypeHint, ArrayImpl> Default for EvalRowMajorDispatcher<TypeHint, ArrayImpl> {
    fn default() -> Self {
        Self {
            _type_hint: std::marker::PhantomData,
        }
    }
}

impl<ArrayImpl, const NDIM: usize> DispatchEvalRowMajor<NDIM>
    for EvalRowMajorDispatcher<Unknown, ArrayImpl>
where
    ArrayImpl::Item: Copy + Default,
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Output = StridedDynArray<ArrayImpl::Item, NDIM>;

    type ArrayImpl = ArrayImpl;

    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output {
        let mut output = StridedDynArray::row_major(arr.shape());
        output.fill_from(arr);
        output
    }
}

impl<ArrayImpl, const NDIM: usize, const N: usize> DispatchEvalRowMajor<NDIM>
    for EvalRowMajorDispatcher<Stack<N>, ArrayImpl>
where
    ArrayImpl::Item: Copy + Default,
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Output = Array<StridedBaseArray<ArrayContainer<ArrayImpl::Item, N>, NDIM>, NDIM>;

    type ArrayImpl = ArrayImpl;

    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output {
        let mut output = Array::new(StridedBaseArray::new(
            ArrayContainer::<ArrayImpl::Item, N>::new(),
            arr.shape(),
            row_major_stride_from_shape(arr.shape()),
        ));
        output.fill_from(arr);
        output
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    /// Iterate through the array by value.
    ///
    /// The iterator always proceeds in column-major order.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue](UnsafeRandom1DAccessByValue)
    /// - [Shape](Shape)
    #[inline(always)]
    fn iter_value(&self) -> ArrayDefaultIteratorByValue<'_, ArrayImpl, NDIM> {
        ArrayDefaultIteratorByValue::new(self)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByRef + Shape<NDIM>,
{
    /// Iterate through the array by reference.
    ///
    /// The iterator always proceeds in column-major order.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByRef](UnsafeRandom1DAccessByRef)
    /// - [Shape](Shape)
    #[inline(always)]
    fn iter_ref(&self) -> ArrayDefaultIteratorByRef<'_, ArrayImpl, NDIM> {
        ArrayDefaultIteratorByRef::new(self)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    /// Iterate through the array by mutable reference.
    ///
    /// The iterator always proceeds in column-major order.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut](UnsafeRandom1DAccessMut)
    /// - [Shape](Shape)
    #[inline(always)]
    fn iter_mut(&mut self) -> ArrayDefaultIteratorMut<'_, ArrayImpl, NDIM> {
        ArrayDefaultIteratorMut::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
{
    /// Iterate through the diagonal of the array by value.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByValue](UnsafeRandomAccessByValue)
    /// - [Shape](Shape)
    #[inline(always)]
    fn diag_iter_value(&self) -> ArrayDiagIteratorByValue<'_, ArrayImpl, NDIM> {
        ArrayDiagIteratorByValue::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByRef<NDIM, Item = Item> + Shape<NDIM>,
{
    /// Iterate through the diagonal of the array by reference.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByRef](UnsafeRandomAccessByRef)
    /// - [Shape](Shape)
    #[inline(always)]
    fn diag_iter_ref(&self) -> ArrayDiagIteratorByRef<'_, ArrayImpl, NDIM> {
        ArrayDiagIteratorByRef::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>,
{
    /// Iterate through the diagonal of the array by mutable reference.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByRef](UnsafeRandomAccessByRef)
    /// - [Shape](Shape)
    #[inline(always)]
    fn diag_iter_mut(&mut self) -> ArrayDiagIteratorMut<'_, ArrayImpl, NDIM> {
        ArrayDiagIteratorMut::new(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
{
    /// Fill from another array.
    ///
    /// #Traits
    /// - [UnsafeRandomAccessMut]
    /// - [Shape]
    #[inline(always)]
    fn fill_from<ArrayImplOther>(&mut self, other: &Array<ArrayImplOther, NDIM>)
    where
        ArrayImplOther: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter_value()) {
            *item = other_item;
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    /// Fill from an iterator.
    ///
    /// Note that the array only fills as many values as the other iterator provides.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    fn fill_from_iter<Iter>(&mut self, iter: Iter)
    where
        Iter: Iterator<Item = ArrayImpl::Item>,
    {
        for (item, other_item) in izip!(self.iter_mut(), iter) {
            *item = other_item;
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: ResizeInPlace<NDIM> + Shape<NDIM> + UnsafeRandom1DAccessMut,
{
    /// Fill from another array and resize if necessary.
    ///
    /// This method is especially useful together with [empty_array].
    ///
    /// #Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    fn fill_from_resize<ArrayImplOther>(&mut self, other: &Array<ArrayImplOther, NDIM>)
    where
        ArrayImplOther: Shape<NDIM> + UnsafeRandom1DAccessByValue,
    {
        self.resize_in_place(other.shape());
        self.fill_from(other);
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    /// Fill array with a given value.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    fn fill_with_value(&mut self, value: ArrayImpl::Item) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
    ArrayImpl::Item: std::iter::Sum,
{
    /// Compute the sum of the diagonal values of the array.
    ///
    /// Note: The Item type must support the trait [std::iter::Sum].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    fn trace(&self) -> ArrayImpl::Item {
        self.diag_iter_value().sum::<Self::Item>()
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<1>,
    ArrayImpl::Item: Conj<Output = Item> + std::iter::Sum,
{
    /// Compute the inner product of two 1d arrays.
    ///
    /// The elements of `other` are taken as conjugate.
    ///
    /// Note: The Item type must support the traits [Conj] and [std::iter::Sum].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    fn inner<ArrayImplOther>(&self, other: &Array<ArrayImplOther, 1>) -> Item
    where
        ArrayImplOther: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    {
        assert_eq!(self.shape(), other.shape());
        izip!(self.iter_value(), other.iter_value())
            .map(|(x, y)| x * y.conj())
            .sum()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: Abs,
    <Item as Abs>::Output: Max,
    <<Item as Abs>::Output as Max>::Output: Default,
{
    /// Compute the maximum absolute value over all elements.
    ///
    /// Note: The item type must support [Abs]. The output of [Abs]
    /// must support [Max] and the output of [Max] must support [Default].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    fn max_abs(&self) -> <<Item as Abs>::Output as Max>::Output {
        self.iter_value()
            .fold(<Self::Output as Default>::default(), |acc, elem| {
                Max::max(acc, elem.abs())
            })
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: Abs,
    <Item as Abs>::Output: std::iter::Sum,
{
    /// Compute the 1-norm of a 1d array.
    ///
    /// Note: The item type must support [Abs] and the output of [Abs]
    /// must support [std::iter::Sum].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    fn norm_1(&self) -> <Item as Abs>::Output {
        self.iter_value().map(|elem| elem.abs()).sum()
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    Item: AbsSquare,
    <Item as AbsSquare>::Output: Sqrt<Output = <Item as AbsSquare>::Output> + std::iter::Sum,
{
    /// Compute the 2-norm of a 1d array.
    ///
    /// Note: The item type must support [AbsSquare]. The output of [AbsSquare]
    /// must support [std::iter::Sum] and [Sqrt].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    fn norm_2(&self) -> <Item as AbsSquare>::Output {
        self.iter_value().map(|elem| elem.abs_square()).sum().sqrt()
    }
}

impl<Item, ArrayImpl, const NDIM: usize> EvaluateObject for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType + UnsafeRandom1DAccessByValue<Item = Item>,
    EvalDispatcher<ArrayImpl::Type, ArrayImpl>: DispatchEval<NDIM, ArrayImpl = ArrayImpl>,
{
    type Output = <EvalDispatcher<ArrayImpl::Type, ArrayImpl> as DispatchEval<NDIM>>::Output;

    /// Evaluate array into a new array.
    ///
    /// The output
    #[inline(always)]
    fn eval(&self) -> Self::Output {
        let dispatcher = EvalDispatcher::<ArrayImpl::Type, ArrayImpl>::default();
        dispatcher.dispatch(self)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> EvaluateRowMajorArray for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType + UnsafeRandom1DAccessByValue<Item = Item>,
    EvalRowMajorDispatcher<ArrayImpl::Type, ArrayImpl>:
        DispatchEvalRowMajor<NDIM, ArrayImpl = ArrayImpl>,
{
    type Output =
        <EvalRowMajorDispatcher<ArrayImpl::Type, ArrayImpl> as DispatchEvalRowMajor<NDIM>>::Output;

    fn eval_row_major(&self) -> Self::Output {
        let dispatcher = EvalRowMajorDispatcher::<ArrayImpl::Type, ArrayImpl>::default();
        dispatcher.dispatch(self)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    fn into_type<T>(
        self,
    ) -> Array<
        ArrayUnaryOperator<ArrayImpl::Item, T, ArrayImpl, fn(ArrayImpl::Item) -> T, NDIM>,
        NDIM,
    >
    where
        ArrayImpl::Item: Into<T>,
    {
        Array::new(ArrayUnaryOperator::new(self, |item| item.into()))
    }
}

impl<ArrayImpl> ColumnIterator for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    type Col<'a>
        = Array<ArraySlice<ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = ColIterator<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn col_iter(&self) -> Self::Iter<'_> {
        ColIterator::new(self)
    }
}

impl<ArrayImpl> ColumnIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    type Col<'a>
        = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = ColIteratorMut<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn col_iter_mut(&mut self) -> Self::Iter<'_> {
        ColIteratorMut::new(self)
    }
}

impl<ArrayImpl> crate::traits::iterators::RowIterator for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    type Row<'a>
        = Array<ArraySlice<ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = crate::dense::array::iterators::RowIterator<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn row_iter(&self) -> Self::Iter<'_> {
        crate::dense::array::iterators::RowIterator::new(self)
    }
}

impl<ArrayImpl> crate::traits::iterators::RowIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    type Row<'a>
        = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>
    where
        Self: 'a;

    type Iter<'a>
        = crate::dense::array::iterators::RowIteratorMut<'a, ArrayImpl, 2>
    where
        Self: 'a;

    fn row_iter_mut(&mut self) -> Self::Iter<'_> {
        crate::dense::array::iterators::RowIteratorMut::new(self)
    }
}

impl<ArrayImpl> AijIteratorByValue for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<2>,
{
    fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], Self::Item)> + '_ {
        let iter = ArrayDefaultIteratorByValue::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> AijIteratorByRef for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByRef + Shape<2>,
{
    fn iter_aij_ref(&self) -> impl Iterator<Item = ([usize; 2], &Self::Item)> + '_ {
        let iter = ArrayDefaultIteratorByRef::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> AijIteratorMut for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<2>,
{
    fn iter_aij_mut(&mut self) -> impl Iterator<Item = ([usize; 2], &mut Self::Item)> + '_ {
        let shape = self.shape();
        let iter = ArrayDefaultIteratorMut::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), shape)
    }
}

impl<ArrayImpl, ArrayImplOther, const NDIM: usize> AddAssign<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    Array<ArrayImpl, NDIM>: SumFrom<Array<ArrayImplOther, NDIM>>,
{
    fn add_assign(&mut self, rhs: Array<ArrayImplOther, NDIM>) {
        self.sum_from(&rhs)
    }
}

impl<Out, ArrayImpl, ArrayImplOther, const NDIM: usize> SubAssign<Array<ArrayImplOther, NDIM>>
    for Array<ArrayImpl, NDIM>
where
    Array<ArrayImplOther, NDIM>: Neg<Output = Out>,
    Array<ArrayImpl, NDIM>: SumFrom<Out>,
{
    fn sub_assign(&mut self, rhs: Array<ArrayImplOther, NDIM>) {
        self.sum_from(&rhs.neg())
    }
}

impl<Item, ArrayImpl, const NDIM: usize> MulAssign<Item> for Array<ArrayImpl, NDIM>
where
    Item: Copy,
    Self: ArrayIteratorMut<Item = Item>,
    Item: MulAssign<Item>,
{
    fn mul_assign(&mut self, rhs: Item) {
        for item in self.iter_mut() {
            *item *= rhs;
        }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Neg for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
    Item: Neg<Output = Item>,
{
    type Output = Array<ArrayNeg<ArrayImpl, NDIM>, NDIM>;

    fn neg(self) -> Self::Output {
        Array::new(ArrayNeg::new(self))
    }
}
