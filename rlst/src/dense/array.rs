//! Basic Array type
//!
//! [Array] is the basic type for dense calculations in Rlst. The full definition
//! `Array<ArrayImpl, NDIM>` represents a tensor with `NDIM` axes implemented through
//! the implementeation type `ArrayImpl`.

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
        data_container::ContainerTypeHint,
    },
    AsOwnedRefType, AsOwnedRefTypeMut, DispatchEval, DispatchEvalRowMajor, FillFrom, Stack,
    Unknown,
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

    /// Create a new heap allocated array by providing a shape and a vector of data.
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

impl<Item: Clone + Default> From<&[Item]> for DynArray<Item, 1> {
    fn from(value: &[Item]) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value.to_vec())
    }
}

impl<Item: Clone + Default> From<Vec<Item>> for DynArray<Item, 1> {
    fn from(value: Vec<Item>) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value)
    }
}

impl<Item: Clone + Default, const NDIM: usize>
    Array<StridedBaseArray<VectorContainer<Item>, NDIM>, NDIM>
{
    /// Create a new heap allocated array from a given shape and stride.
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

impl<Item: Clone + Default, const NDIM: usize>
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

impl<ArrayImpl, const NDIM: usize> ContainerTypeHint for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerTypeHint,
{
    type TypeHint = ArrayImpl::TypeHint;
}

impl<ArrayImpl, const NDIM: usize> BaseItem for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem,
{
    type Item = ArrayImpl::Item;
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Shape<NDIM> for Array<ArrayImpl, NDIM> {
    #[inline(always)]
    fn shape(&self) -> [usize; NDIM] {
        self.0.shape()
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Len for Array<ArrayImpl, NDIM> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }
}

impl<ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> UnsafeRandomAccessByValue<NDIM>
    for Array<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item {
        self.0.get_value_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> UnsafeRandomAccessByRef<NDIM>
    for Array<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item {
        self.0.get_unchecked(multi_index)
    }
}

impl<ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> UnsafeRandomAccessMut<NDIM>
    for Array<ArrayImpl, NDIM>
{
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
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.0.data()
    }
}

impl<ArrayImpl: RawAccessMut, const NDIM: usize> RawAccessMut for Array<ArrayImpl, NDIM> {
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

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> UnsafeRandom1DAccessByValue
    for Array<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item {
        self.0.get_value_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> UnsafeRandom1DAccessByRef
    for Array<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item {
        self.0.get_1d_unchecked(index)
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> UnsafeRandom1DAccessMut
    for Array<ArrayImpl, NDIM>
{
    #[inline(always)]
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.0.get_1d_unchecked_mut(index)
    }
}

impl<Item, const NDIM: usize> DynArray<Item, NDIM>
where
    Item: Clone + Default,
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
