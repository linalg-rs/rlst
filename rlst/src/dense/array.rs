//! Basic Array type
//!
//! [Array] is the basic type for dense calculations in Rlst. The full definition
//! `Array<ArrayImpl, NDIM>` represents a tensor with `NDIM` axes implemented through
//! the implementeation type `ArrayImpl`.

use empty_axis::AxisPosition;
use iterators::{
    ArrayDefaultIteratorByRef, ArrayDefaultIteratorByValue, ArrayDefaultIteratorMut,
    ArrayDiagIteratorByRef, ArrayDiagIteratorByValue, ArrayDiagIteratorMut, ColIterator,
    ColIteratorMut, RowIterator, RowIteratorMut,
};
use itertools::izip;
use num::{One, Zero, traits::MulAdd};
use operators::{
    addition::ArrayAddition, cast::ArrayCast, cmp_wise_division::CmpWiseDivision,
    cmp_wise_product::CmpWiseProduct, coerce::CoerceArray, mul_add::MulAddImpl, negation::ArrayNeg,
    reverse_axis::ReverseAxis, scalar_mult::ArrayScalarMult, subtraction::ArraySubtraction,
    transpose::ArrayTranspose, unary_op::ArrayUnaryOperator,
};

use paste::paste;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal, StandardUniform};
use reference::{ArrayRef, ArrayRefMut};
use slice::ArraySlice;
use subview::ArraySubView;

use crate::{
    AsMatrixApply, AsMultiIndex, AsOwnedRefType, AsOwnedRefTypeMut, DispatchEval,
    DispatchEvalRowMajor, EvaluateObject, EvaluateRowMajorArray, Gemm, IsGreaterByOne,
    IsGreaterZero, Max, MemoryLayout, NumberType, RandScalar, RandomAccessByValue, RlstError,
    RlstResult, RlstScalar, Stack, TransMode, Unknown,
    base_types::{c32, c64},
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
        base_operations::{BaseItem, ResizeInPlace, Shape, Stride},
        data_container::ContainerType,
    },
};

use super::{data_container::ArrayContainer, layout::row_major_stride_from_shape};

pub mod empty_axis;
pub mod iterators;
pub mod mult_into;
pub mod operations;
pub mod operators;
// pub mod rank1_array;
pub mod flattened;
pub mod reference;
pub mod slice;
pub mod subview;

/// A basic dynamically allocated array.
pub type DynArray<Item, const NDIM: usize> = Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM>;

/// A basic dynamically allocated array with a given stride.
pub type StridedDynArray<Item, const NDIM: usize> =
    Array<StridedBaseArray<VectorContainer<Item>, NDIM>, NDIM>;

/// A statically allocated array.
///
/// `N` is the number of elements to be reserved in the static
/// container.
pub type StaticArray<Item, const NDIM: usize, const N: usize> =
    Array<BaseArray<ArrayContainer<Item, N>, NDIM>, NDIM>;

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
    pub fn imp(&self) -> &ArrayImpl {
        &self.0
    }

    /// Return a mutable reference to the implementation.
    #[inline(always)]
    pub fn imp_mut(&mut self) -> &mut ArrayImpl {
        &mut self.0
    }

    /// Extract the inner implementation.
    #[inline(always)]
    pub fn into_imp(self) -> ArrayImpl {
        self.0
    }

    /// Create an owned reference to the array.
    #[inline(always)]
    pub fn r(&self) -> Array<ArrayRef<'_, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayRef::new(self))
    }

    /// Create an owned mutable reference to the array.
    #[inline(always)]
    pub fn r_mut(&mut self) -> Array<ArrayRefMut<'_, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayRefMut::new(self))
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

impl<Item: Copy + Default> DynArray<Item, 2> {
    /// Create a new dense matrix with shape `shape` filled with values from the iterator `iter`.
    pub fn from_iter_aij<Iter: Iterator<Item = ([usize; 2], Item)>>(
        shape: [usize; 2],
        iter: Iter,
    ) -> DynArray<Item, 2> {
        let mut out = DynArray::<Item, 2>::from_shape(shape);

        for (index, item) in iter {
            out[index] = item;
        }

        out
    }
}

impl<Item: Copy + Default> From<&[Item]> for DynArray<Item, 1> {
    fn from(value: &[Item]) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value.to_vec())
    }
}

impl<Item: Copy + Default, const N: usize> From<[Item; N]> for StaticArray<Item, 1, N> {
    fn from(value: [Item; N]) -> Self {
        StaticArray::new(BaseArray::new(value.into(), [N]))
    }
}

impl<Item: Copy + Default> From<Vec<Item>> for DynArray<Item, 1> {
    fn from(value: Vec<Item>) -> Self {
        DynArray::<Item, 1>::from_shape_and_vec([value.len()], value)
    }
}

impl<Item: Copy + Default> From<Vec<Vec<Item>>> for DynArray<Item, 2> {
    fn from(value: Vec<Vec<Item>>) -> Self {
        let nrows = value.len();
        if nrows == 0 {
            empty_array()
        } else {
            let ncols = value.first().unwrap().len();
            // Check that all columns have identical length

            for row in value.iter() {
                assert_eq!(ncols, row.len(), "All rows must have equal length.");
            }

            // Now create the matrix and fill with values.

            let mut out = DynArray::<Item, 2>::from_shape([nrows, ncols]);

            for (row_index, row) in value.iter().enumerate() {
                for (col_index, &elem) in row.iter().enumerate() {
                    out[[row_index, col_index]] = elem;
                }
            }

            out
        }
    }
}

impl<Item: Copy + Default, const NDIM: usize> StridedDynArray<Item, NDIM> {
    /// Create a new heap allocated array from a given `shape` and `stride`.
    #[inline(always)]
    pub fn from_shape_and_stride(shape: [usize; NDIM], stride: [usize; NDIM]) -> Self {
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
    /// - [Shape]
    #[inline(always)]
    pub fn shape(&self) -> [usize; NDIM] {
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
    /// - [Shape]
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Return true if the array is empty.
    ///
    /// This is equivalent to at least one dimension being zero.
    /// # Traits:
    /// - [Shape]
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
    /// - [UnsafeRandomAccessByValue]
    #[inline(always)]
    pub unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> ArrayImpl::Item {
        unsafe { self.0.get_value_unchecked(multi_index) }
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
    /// - [UnsafeRandomAccessByRef]
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &ArrayImpl::Item {
        unsafe { self.0.get_unchecked(multi_index) }
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
    /// - [UnsafeRandomAccessMut]
    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut ArrayImpl::Item {
        unsafe { self.0.get_unchecked_mut(multi_index) }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: RandomAccessByValue<NDIM>,
{
    /// Get an element by value.
    ///
    /// # Traits
    /// - [RandomAccessByValue]
    #[inline(always)]
    pub fn get_value(&self, multi_index: [usize; NDIM]) -> Option<ArrayImpl::Item> {
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
    /// - [RandomAccessByRef]
    #[inline(always)]
    pub fn get(&self, multi_index: [usize; NDIM]) -> Option<&ArrayImpl::Item> {
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
    /// - [RandomAccessMut]
    #[inline(always)]
    pub fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut ArrayImpl::Item> {
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
    /// - [RawAccess]
    #[inline(always)]
    pub fn data(&self) -> Option<&[ArrayImpl::Item]> {
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
    /// - [RawAccessMut]
    #[inline(always)]
    pub fn data_mut(&mut self) -> Option<&mut [ArrayImpl::Item]> {
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
    /// - [Stride]
    #[inline(always)]
    pub fn stride(&self) -> [usize; NDIM] {
        self.0.stride()
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Stride<NDIM> + Shape<NDIM>,
{
    /// Return the memory layout
    ///
    /// The possible memory layouts are defined in [MemoryLayout].
    #[inline(always)]
    pub fn memory_layout(&self) -> MemoryLayout {
        self.0.memory_layout()
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
    /// - [ResizeInPlace]
    pub fn resize_in_place(&mut self, shape: [usize; NDIM]) {
        self.0.resize_in_place(shape)
    }
}

/// Create an empty array of given type and dimension.
///
/// Empty arrays serve as convenient containers for input into functions that
/// resize an array before filling it with data.
pub fn empty_array<Item, const NDIM: usize>() -> Array<BaseArray<VectorContainer<Item>, NDIM>, NDIM>
where
    Item: Copy + Default,
{
    let shape = [0; NDIM];
    let container = VectorContainer::new(0);
    Array::new(BaseArray::new(container, shape))
}

impl<'a, Item, const NDIM: usize> SliceArray<'a, Item, NDIM>
where
    Item: Copy + Default,
{
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

impl<'a, Item, const NDIM: usize> SliceArrayMut<'a, Item, NDIM>
where
    Item: Copy + Default,
{
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

impl<'a, Item, const NDIM: usize> StridedSliceArray<'a, Item, NDIM>
where
    Item: Copy + Default,
{
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

impl<'a, Item, const NDIM: usize> StridedSliceArrayMut<'a, Item, NDIM>
where
    Item: Copy + Default,
{
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
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let mut output = empty_array::<Item, NDIM>();
        output.fill_from_resize(arr);
        output
    }
}

impl<Item, const NDIM: usize> StridedDynArray<Item, NDIM>
where
    Item: Copy + Default,
{
    /// Create a new strided array with `stride` and fill with values from `arr`.
    pub fn new_from<ArrayImpl>(stride: [usize; NDIM], arr: &Array<ArrayImpl, NDIM>) -> Self
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let mut output = StridedDynArray::from_shape_and_stride(arr.shape(), stride);
        output.fill_from(arr);
        output
    }

    /// Create a new row-major array from existing array `arr`.
    pub fn row_major_from<ArrayImpl>(arr: &Array<ArrayImpl, NDIM>) -> Self
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        StridedDynArray::new_from(row_major_stride_from_shape(arr.shape()), arr)
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

impl<Item, ArrayImpl, const NDIM: usize> EvaluateObject for Array<ArrayImpl, NDIM>
where
    ArrayImpl: ContainerType + UnsafeRandom1DAccessByValue<Item = Item>,
    EvalDispatcher<ArrayImpl::Type, ArrayImpl>: DispatchEval<NDIM, ArrayImpl = ArrayImpl>,
{
    type Output = <EvalDispatcher<ArrayImpl::Type, ArrayImpl> as DispatchEval<NDIM>>::Output;

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
    /// Iterate through the array by value.
    ///
    /// The iterator always proceeds in column-major order.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn iter_value(&self) -> ArrayDefaultIteratorByValue<'_, ArrayImpl, NDIM> {
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
    /// - [UnsafeRandom1DAccessByRef]
    /// - [Shape]
    #[inline(always)]
    pub fn iter_ref(&self) -> ArrayDefaultIteratorByRef<'_, ArrayImpl, NDIM> {
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
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    pub fn iter_mut(&mut self) -> ArrayDefaultIteratorMut<'_, ArrayImpl, NDIM> {
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
    /// - [UnsafeRandomAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn diag_iter_value(&self) -> ArrayDiagIteratorByValue<'_, ArrayImpl, NDIM> {
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
    /// - [UnsafeRandomAccessByRef]
    /// - [Shape]
    #[inline(always)]
    pub fn diag_iter_ref(&self) -> ArrayDiagIteratorByRef<'_, ArrayImpl, NDIM> {
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
    /// - [UnsafeRandomAccessByRef]
    /// - [Shape]
    #[inline(always)]
    pub fn diag_iter_mut(&mut self) -> ArrayDiagIteratorMut<'_, ArrayImpl, NDIM> {
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
    pub fn fill_from<ArrayImplOther>(&mut self, other: &Array<ArrayImplOther, NDIM>)
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
    pub fn fill_from_iter<Iter>(&mut self, iter: Iter)
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
    pub fn fill_from_resize<ArrayImplOther>(&mut self, other: &Array<ArrayImplOther, NDIM>)
    where
        ArrayImplOther: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = ArrayImpl::Item>,
    {
        self.resize_in_place(other.shape());
        self.fill_from(other);
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    /// Fill array with a given value.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    #[inline(always)]
    pub fn fill_with_value(&mut self, value: ArrayImpl::Item) {
        for item in self.iter_mut() {
            *item = value;
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    /// Set the elements of the array to zero.
    pub fn set_zero(&mut self) {
        for item in self.iter_mut() {
            *item = Default::default();
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + UnsafeRandomAccessMut<NDIM> + Shape<NDIM>,
    ArrayImpl::Item: One,
{
    /// Set all off-diagonal elements to zero and the diagonal to one.
    pub fn set_identity(&mut self) {
        self.set_zero();
        self.diag_iter_mut().for_each(|elem| *elem = One::one());
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>,
    ArrayImpl::Item: std::iter::Sum,
{
    /// Compute the sum of the diagonal values of the array.
    ///
    /// Note: The Item type must support the trait [std::iter::Sum].
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn trace(&self) -> ArrayImpl::Item {
        self.diag_iter_value().sum()
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    ArrayImpl::Item: Conj<Output = Item>
        + std::ops::Mul<Output = ArrayImpl::Item>
        + std::ops::Add<Output = Item>,
{
    /// Compute the inner product of two 1d arrays.
    ///
    /// The elements of `other` are taken as conjugate.
    ///
    /// Returns `None` if the arrays are empty.
    ///
    /// Note: The Item type must support the traits [Conj], [std::iter::Sum] and [std::ops::Mul].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn inner<ArrayImplOther>(&self, other: &Array<ArrayImplOther, 1>) -> Option<Item>
    where
        ArrayImplOther: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    {
        assert_eq!(self.shape(), other.shape());
        izip!(self.iter_value(), other.iter_value())
            .map(|(x, y)| x * y.conj())
            .reduce(std::ops::Add::add)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: Abs,
    <Item as Abs>::Output: Max<Output = <Item as Abs>::Output>,
{
    /// Compute the maximum absolute value over all elements.
    ///
    /// Note: The item type must support [Abs] and the output of [Abs]
    /// must support [Max].
    ///
    /// The function returns `None` if the array is empty.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn max_abs(&self) -> Option<<Item as Abs>::Output> {
        self.iter_value().map(|elem| elem.abs()).reduce(Max::max)
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    Item: Abs,
    <Item as Abs>::Output: std::ops::Add<Output = <Item as Abs>::Output>,
{
    /// Compute the 1-norm of a 1d array.
    ///
    /// The function returns `None` if the array is empty.
    ///
    /// Note: The item type must support [Abs] and the output of [Abs] must
    /// support [std::ops::Add].
    ///
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn norm_1(&self) -> Option<<Item as Abs>::Output> {
        self.iter_value()
            .map(|elem| elem.abs())
            .reduce(std::ops::Add::add)
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 1>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<1>,
    Item: AbsSquare,
    <Item as AbsSquare>::Output: Sqrt<Output = <Item as AbsSquare>::Output>
        + std::ops::Add<Output = <Item as AbsSquare>::Output>,
{
    /// Compute the 2-norm of a 1d array.
    ///
    /// Note: The item type must support [AbsSquare]. The output of [AbsSquare]
    /// must support [std::ops::Add] and [Sqrt].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn norm_2(&self) -> Option<<Item as AbsSquare>::Output> {
        self.iter_value()
            .map(|elem| elem.abs_square())
            .reduce(std::ops::Add::add)
            .map(|elem| elem.sqrt())
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: AbsSquare,
    <Item as AbsSquare>::Output: Sqrt<Output = <Item as AbsSquare>::Output>
        + std::ops::Add<Output = <Item as AbsSquare>::Output>
        + Copy
        + Default,
{
    /// Compute the Frobenius-Norm of a nd array.
    ///
    /// Note: The item type must support [AbsSquare]. The output of [AbsSquare]
    /// must support [std::ops::Add], [Sqrt], [Copy], and [Default].
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[inline(always)]
    pub fn norm_fro(&self) -> Option<<Item as AbsSquare>::Output> {
        self.r()
            .abs_square()
            .iter_value()
            .reduce(std::ops::Add::add)
            .map(|elem| elem.sqrt())
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    /// Convert an array into the new item type T.
    ///
    /// Note: It is required that `ArrayImpl::Item: Into<T>`.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    #[allow(clippy::type_complexity)]
    pub fn into_type<T>(
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

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    /// Return a column iterator for a 2d array.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByValue]
    /// - [Shape]
    pub fn col_iter(&self) -> ColIterator<'_, ArrayImpl, 2> {
        ColIterator::new(self)
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessMut<2>,
{
    /// Return a mutable column iterator for a 2d array.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessMut]
    /// - [Shape]
    pub fn col_iter_mut(&mut self) -> ColIteratorMut<'_, ArrayImpl, 2> {
        ColIteratorMut::new(self)
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    /// Return a row iterator for a 2d array.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessByValue]
    /// - [Shape]
    pub fn row_iter(&self) -> RowIterator<'_, ArrayImpl, 2> {
        RowIterator::new(self)
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + UnsafeRandomAccessByValue<2>,
{
    /// Return a mutable row iterator for a 2d array.
    ///
    /// # Traits
    /// - [UnsafeRandomAccessMut]
    /// - [Shape]
    pub fn row_iter_mut(&mut self) -> RowIteratorMut<'_, ArrayImpl, 2> {
        RowIteratorMut::new(self)
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<2>,
{
    /// Return an iterator of the form `(i, j, data)`.
    ///
    /// Here, `i` is the row, `j` is the column, and `data` is the associated element
    /// returned by value.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByValue]
    /// - [Shape]
    pub fn iter_aij_value(&self) -> impl Iterator<Item = ([usize; 2], ArrayImpl::Item)> + '_ {
        let iter = ArrayDefaultIteratorByValue::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByRef + Shape<2>,
{
    /// Return an iterator of the form `(i, j, &data)`.
    ///
    /// Here, `i` is the row, `j` is the column, and `&data` is the reference to the associated
    /// element.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByRef]
    /// - [Shape]
    pub fn iter_aij_ref(&self) -> impl Iterator<Item = ([usize; 2], &ArrayImpl::Item)> + '_ {
        let iter = ArrayDefaultIteratorByRef::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), self.shape())
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<2>,
{
    /// Return an iterator of the form `(i, j, &mut data)`.
    ///
    /// Here, `i` is the row, `j` is the column, and `&data` is the mutable reference to the associated
    /// element.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessByRef]
    /// - [Shape]
    pub fn iter_aij_mut(
        &mut self,
    ) -> impl Iterator<Item = ([usize; 2], &mut ArrayImpl::Item)> + '_ {
        let shape = self.shape();
        let iter = ArrayDefaultIteratorMut::new(self);
        AsMultiIndex::multi_index(std::iter::Iterator::enumerate(iter), shape)
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Cast array to type `T`.
    ///
    /// The cast is done through [num::cast::cast] and source and target types need to
    /// support casting through that function.
    pub fn cast<Target>(self) -> Array<ArrayCast<Target, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayCast::new(self))
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Coerce the array to a specific dimension.
    ///
    /// This is useful to coerce from a generic dimension parameter to
    /// a specific number of dimensions.
    pub fn coerce_dim<const CDIM: usize>(
        self,
    ) -> RlstResult<Array<CoerceArray<ArrayImpl, NDIM, CDIM>, CDIM>> {
        if CDIM == NDIM {
            // If the dimensions and item types match return a CoerceArray
            Ok(Array::new(CoerceArray::new(self)))
        } else {
            // Otherwise, we need to coerce the array.
            Err(RlstError::GeneralError(
                format!("Cannot coerce array: dimensions do not match {CDIM} != {NDIM}.")
                    .to_string(),
            ))
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Permute axes of an array.
    ///
    /// The `permutation` gives the new ordering of the axes.
    pub fn permute_axes(
        self,
        permutation: [usize; NDIM],
    ) -> Array<ArrayTranspose<ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayTranspose::new(self, permutation))
    }

    /// Transpose an array.
    ///
    /// The transpose of an n-dimensional array reverses the order of the axes.
    pub fn transpose(self) -> Array<ArrayTranspose<ArrayImpl, NDIM>, NDIM> {
        let mut permutation = [0; NDIM];

        for (ind, p) in (0..NDIM).rev().enumerate() {
            permutation[ind] = p;
        }

        self.permute_axes(permutation)
    }
}

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    Item: RandScalar + RlstScalar,
    ArrayImpl: UnsafeRandom1DAccessMut<Item = Item> + Shape<NDIM>,
    StandardNormal: Distribution<Item::Real>,
    StandardUniform: Distribution<Item::Real>,
{
    /// Fill an array with normally distributed random numbers.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    /// - Item: [RandScalar] + [RlstScalar]
    pub fn fill_from_standard_normal<R: Rng>(&mut self, rng: &mut R) {
        let dist = StandardNormal;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }

    /// Fill an array with equally distributed random numbers.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    /// - Item: [RandScalar] + [RlstScalar]
    pub fn fill_from_equally_distributed<R: Rng>(&mut self, rng: &mut R) {
        let dist = StandardUniform;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }

    /// Fill an array with equally distributed random numbers using a given `seed`.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    /// - Item: [RandScalar] + [RlstScalar]
    pub fn fill_from_seed_equally_distributed(&mut self, seed: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let dist = StandardUniform;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(&mut rng, &dist));
    }

    /// Fill an array with normally distributed random numbers using a given `seed`.
    ///
    /// # Traits
    /// - [UnsafeRandom1DAccessMut]
    /// - [Shape]
    /// - Item: [RandScalar] + [RlstScalar]
    pub fn fill_from_seed_normally_distributed(&mut self, seed: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let dist = StandardNormal;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(&mut rng, &dist));
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Reverse a single `axis` of the array.
    pub fn reverse_axis(self, axis: usize) -> Array<ReverseAxis<ArrayImpl, NDIM>, NDIM> {
        assert!(axis < NDIM, "Axis out of bounds");
        Array::new(ReverseAxis::new(self, axis))
    }
}

impl<ArrayImpl, const ADIM: usize> Array<ArrayImpl, ADIM>
where
    ArrayImpl: Shape<ADIM>,
{
    /// Create a slice from a given array.
    ///
    /// Consider an array `arr` with shape `[a0, a1, a2, a3, ...]`. The function call
    /// `arr.slice(2, 3)` returns a one dimension smaller array indexed by `[a0, a1, 3, a3, ...]`.
    /// Hence, the dimension `2` has been fixed to always have the value `3.`
    ///
    /// # Examples
    ///
    /// If `arr` is a matrix then the first column of the matrix is obtained from
    /// `arr.slice(1, 0)`, while the third row of the matrix is obtained from
    /// `arr.slice(0, 2)`.
    pub fn slice<const NDIM: usize>(
        self,
        axis: usize,
        index: usize,
    ) -> Array<ArraySlice<ArrayImpl, ADIM, NDIM>, NDIM>
    where
        NumberType<ADIM>: IsGreaterByOne<NDIM>,
        NumberType<NDIM>: IsGreaterZero,
    {
        Array::new(ArraySlice::new(self, [axis, index]))
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Return the row with index `row_index` from a two-dimensional array.
    pub fn row(self, row_index: usize) -> Array<ArraySlice<ArrayImpl, 2, 1>, 1> {
        self.slice::<1>(0, row_index)
    }

    /// Return the column with index `col_index` from a two-dimensional array.
    pub fn col(self, col_index: usize) -> Array<ArraySlice<ArrayImpl, 2, 1>, 1> {
        self.slice::<1>(1, col_index)
    }
}

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Move the array into a subview specified by an `offset` and `shape` of the subview.
    ///
    /// The `offset` is the starting index of the subview and the `shape` is the number of indices
    /// in each dimension of the subview.
    pub fn into_subview(
        self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<ArraySubView<ArrayImpl, NDIM>, NDIM> {
        Array::new(ArraySubView::new(self, offset, shape))
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> MulAdd<Item, Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: BaseItem<Item = Item> + Shape<NDIM>,
    ArrayImpl2: BaseItem<Item = Item> + Shape<NDIM>,
    Item: MulAdd<Output = Item> + Copy,
{
    type Output = Array<MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>;
    /// Compentwie form `self * a + b`, where `a` is a scalar and `b` is another array.
    /// The implementation depdends on the `MulAdd` trait from the `num` crate for the componets of
    /// the arrays.
    fn mul_add(
        self,
        a: Item,
        b: Array<ArrayImpl2, NDIM>,
    ) -> Array<MulAddImpl<ArrayImpl1, ArrayImpl2, Item, NDIM>, NDIM>
    where
        ArrayImpl1: BaseItem<Item = Item> + Shape<NDIM>,
        ArrayImpl2: BaseItem<Item = Item> + Shape<NDIM>,
        Item: MulAdd<Output = Item> + Copy,
    {
        Array::new(MulAddImpl::new(self, b, a))
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Add<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    type Output = Array<ArrayAddition<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    #[inline(always)]
    fn add(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(ArrayAddition::new(self, rhs))
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::AddAssign<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    ArrayImpl2: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: std::ops::AddAssign,
{
    #[inline(always)]
    fn add_assign(&mut self, rhs: Array<ArrayImpl2, NDIM>) {
        assert_eq!(self.shape(), rhs.shape());
        for (item1, item2) in izip!(self.iter_mut(), rhs.iter_value()) {
            *item1 += item2;
        }
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Sub<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    type Output = Array<ArraySubtraction<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn sub(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(ArraySubtraction::new(self, rhs))
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::SubAssign<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    ArrayImpl2: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: std::ops::SubAssign,
{
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Array<ArrayImpl2, NDIM>) {
        assert_eq!(self.shape(), rhs.shape());
        for (item1, item2) in izip!(self.iter_mut(), rhs.iter_value()) {
            *item1 -= item2;
        }
    }
}

impl<Item, ArrayImpl, const NDIM: usize> std::ops::Neg for Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
    Item: std::ops::Neg<Output = Item>,
{
    type Output = Array<ArrayNeg<ArrayImpl, NDIM>, NDIM>;

    fn neg(self) -> Self::Output {
        Array::new(ArrayNeg::new(self))
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Div<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    type Output = Array<CmpWiseDivision<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn div(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(CmpWiseDivision::new(self, rhs))
    }
}

impl<Item, ArrayImpl, const NDIM: usize> std::ops::Div<Item> for Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM> + BaseItem<Item = Item>,
    Item: Recip<Output = Item> + std::ops::Div<Output = Item>,
{
    type Output = Array<ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM>;

    fn div(self, rhs: Item) -> Self::Output {
        Array::new(ArrayScalarMult::new(rhs.recip(), self))
    }
}

impl<Item, ArrayImpl, const NDIM: usize> std::ops::DivAssign<Item> for Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    Item: std::ops::DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: Item) {
        for item in self.iter_mut() {
            *item /= rhs;
        }
    }
}

impl<ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::Mul<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM>,
    ArrayImpl2: Shape<NDIM>,
{
    type Output = Array<CmpWiseProduct<ArrayImpl1, ArrayImpl2, NDIM>, NDIM>;

    fn mul(self, rhs: Array<ArrayImpl2, NDIM>) -> Self::Output {
        Array::new(CmpWiseProduct::new(self, rhs))
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::MulAssign<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    ArrayImpl2: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: std::ops::MulAssign,
{
    fn mul_assign(&mut self, rhs: Array<ArrayImpl2, NDIM>) {
        for (item1, item2) in izip!(self.iter_mut(), rhs.iter_value()) {
            *item1 *= item2;
        }
    }
}

impl<Item, ArrayImpl1, ArrayImpl2, const NDIM: usize> std::ops::DivAssign<Array<ArrayImpl2, NDIM>>
    for Array<ArrayImpl1, NDIM>
where
    ArrayImpl1: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    ArrayImpl2: Shape<NDIM> + UnsafeRandom1DAccessByValue<Item = Item>,
    Item: std::ops::DivAssign,
{
    fn div_assign(&mut self, rhs: Array<ArrayImpl2, NDIM>) {
        for (item1, item2) in izip!(self.iter_mut(), rhs.iter_value()) {
            *item1 /= item2;
        }
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

        impl<ArrayImpl, const NDIM: usize> std::ops::Mul<$ScalarType> for Array<ArrayImpl, NDIM>
        where
            ArrayImpl: Shape<NDIM> + BaseItem<Item = $ScalarType>,
        {
            type Output = Array<ArrayScalarMult<$ScalarType, ArrayImpl, NDIM>, NDIM>;

            fn mul(self, rhs: $ScalarType) -> Self::Output {
                Array::new(ArrayScalarMult::new(rhs, self))
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

impl<Item, ArrayImpl, const NDIM: usize> std::ops::MulAssign<Item> for Array<ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessMut<Item = Item>,
    Item: std::ops::MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: Item) {
        for item in self.iter_mut() {
            *item *= rhs;
        }
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Create a new array by applying the unitary operator `op` to each element of `self`.
    pub fn unary_op<OpItem, OpTarget, Op: Fn(OpItem) -> OpTarget>(
        self,
        op: Op,
    ) -> Array<ArrayUnaryOperator<OpItem, OpTarget, ArrayImpl, Op, NDIM>, NDIM> {
        Array::new(ArrayUnaryOperator::new(self, op))
    }
}

impl<Item: Default + Copy + std::ops::Mul<Output = Item>, ArrayImpl, const NDIM: usize>
    Array<ArrayImpl, NDIM>
where
    ArrayImpl: BaseItem<Item = Item>,
{
    /// Multiple the array with a given `scalar`.
    ///
    /// Note: The `Item` type must support [std::ops::Mul].
    ///
    pub fn scalar_mul(self, scalar: Item) -> Array<ArrayScalarMult<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayScalarMult::new(scalar, self))
    }
}

macro_rules! impl_unary_op_trait {
    ($name:ident, $method_name:ident) => {
        paste! {

        use crate::traits::number_traits::$name;
        impl<Item: $name, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
            where
                ArrayImpl: BaseItem<Item = Item>,
            {
                #[inline(always)]
                #[doc = "Componentwise apply `" $method_name "` to the array."]
                pub fn $method_name(self) -> Array<ArrayUnaryOperator<Item, <Item as $name>::Output, ArrayImpl, fn(Item) -> <Item as $name>::Output, NDIM>, NDIM> {
                    self.unary_op(|x| x.$method_name())
                }
            }
        }

    };
}

impl_unary_op_trait!(Conj, conj);
impl_unary_op_trait!(Abs, abs);
impl_unary_op_trait!(Square, square);
impl_unary_op_trait!(AbsSquare, abs_square);
impl_unary_op_trait!(Sqrt, sqrt);
impl_unary_op_trait!(Exp, exp);
impl_unary_op_trait!(Ln, ln);
impl_unary_op_trait!(Recip, recip);
impl_unary_op_trait!(Sin, sin);
impl_unary_op_trait!(Cos, cos);
impl_unary_op_trait!(Tan, tan);
impl_unary_op_trait!(Asin, asin);
impl_unary_op_trait!(Acos, acos);
impl_unary_op_trait!(Atan, atan);
impl_unary_op_trait!(Sinh, sinh);
impl_unary_op_trait!(Cosh, cosh);
impl_unary_op_trait!(Tanh, tanh);
impl_unary_op_trait!(Asinh, asinh);
impl_unary_op_trait!(Acosh, acosh);
impl_unary_op_trait!(Atanh, atanh);

impl<Item, ArrayImplX, ArrayImplY, ArrayImpl>
    AsMatrixApply<Array<ArrayImplX, 2>, Array<ArrayImplY, 2>> for Array<ArrayImpl, 2>
where
    Item: Gemm + Copy,
    ArrayImplX: RawAccess<Item = Item> + Stride<2> + Shape<2>,
    ArrayImplY: RawAccessMut<Item = Item> + Stride<2> + Shape<2>,
    ArrayImpl: RawAccess<Item = Item> + Stride<2> + Shape<2>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &Array<ArrayImplX, 2>,
        beta: Self::Item,
        y: &mut Array<ArrayImplY, 2>,
    ) {
        let stride_a = self.stride();
        let stride_x = x.stride();
        let stride_y = y.stride();

        let shape_a = self.shape();
        let shape_x = x.shape();
        let shape_y = y.shape();

        assert_eq!(
            shape_a[0], shape_y[0],
            "`y` has incompatible shape {shape_y:#?} with `self` {shape_a:#?}."
        );

        assert_eq!(
            shape_a[1], shape_x[0],
            "`x` has incompatible shape {shape_x:#?} with `self` {shape_a:#?}."
        );

        assert_eq!(
            shape_y[1], shape_x[1],
            "`x` has incompatible shape {shape_x:#?} with `y` {shape_y:#?}."
        );

        let m = shape_a[0];
        let n = shape_x[1];
        let k = shape_a[1];

        <Item as Gemm>::gemm(
            TransMode::NoTrans,
            TransMode::NoTrans,
            m,
            n,
            k,
            alpha,
            self.data().unwrap(),
            stride_a[0],
            stride_a[1],
            x.data().unwrap(),
            stride_x[0],
            stride_x[1],
            beta,
            y.data_mut().unwrap(),
            stride_y[0],
            stride_y[1],
        );
    }
}

impl<Item, ArrayImplX, ArrayImplY, ArrayImpl>
    AsMatrixApply<Array<ArrayImplX, 1>, Array<ArrayImplY, 1>> for Array<ArrayImpl, 2>
where
    Item: Gemm + Copy,
    ArrayImplX: RawAccess<Item = Item> + Stride<1> + Shape<1>,
    ArrayImplY: RawAccessMut<Item = Item> + Stride<1> + Shape<1>,
    ArrayImpl: RawAccess<Item = Item> + Stride<2> + Shape<2>,
{
    fn apply(
        &self,
        alpha: Self::Item,
        x: &Array<ArrayImplX, 1>,
        beta: Self::Item,
        y: &mut Array<ArrayImplY, 1>,
    ) {
        self.apply(
            alpha,
            &x.r().insert_empty_axis::<2>(AxisPosition::Back),
            beta,
            &mut y.r_mut().insert_empty_axis::<2>(AxisPosition::Back),
        );
    }
}

impl<Item, ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: RawAccess<Item = Item> + Stride<2> + Shape<2>,
    Item: Gemm + Copy + Default + One + Zero,
{
    /// Compute the matrix-matrix product of `self` with `other`.
    pub fn dot<ArrayImplOther, const NDIM: usize>(
        &self,
        other: &Array<ArrayImplOther, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        ArrayImplOther: RawAccess<Item = Item> + Shape<NDIM> + Stride<NDIM>,
    {
        let a_shape = self.shape();
        let x_shape = other.shape();

        let mut out = empty_array::<Item, NDIM>();

        if NDIM == 1 {
            let mut out = out.r_mut().coerce_dim::<1>().unwrap();
            out.resize_in_place([a_shape[0]]);

            self.apply(
                Item::one(),
                &other.r().coerce_dim::<1>().unwrap(),
                Item::zero(),
                &mut out,
            );
        } else if NDIM == 2 {
            let mut out = out.r_mut().coerce_dim::<2>().unwrap();
            out.resize_in_place([a_shape[0], x_shape[1]]);

            self.apply(
                Item::one(),
                &other.r().coerce_dim::<2>().unwrap(),
                Item::zero(),
                &mut out,
            )
        } else {
            panic!("NDIM = {NDIM} not supported.");
        }

        out
    }
}

#[cfg(test)]
mod test {

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    pub fn test_add() {
        let a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![2.5, 4.3, 7.0].into();

        crate::assert_array_relative_eq!((a.r() + b.r()).eval(), expected, 1E-10);
    }

    #[test]
    pub fn test_add_assign() {
        let mut a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![2.5, 4.3, 7.0].into();

        a += b;

        crate::assert_array_relative_eq!(a, expected, 1E-10);
    }

    #[test]
    pub fn test_sub() {
        let a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![-0.5, -0.3, -1.0].into();

        crate::assert_array_relative_eq!((a.r() - b.r()).eval(), expected, 1E-10);
    }

    #[test]
    pub fn test_sub_assign() {
        let mut a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![-0.5, -0.3, -1.0].into();

        a -= b;

        crate::assert_array_relative_eq!(a, expected, 1E-10);
    }

    #[test]
    pub fn test_cmp_mul() {
        let a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![1.5, 4.6, 12.0].into();

        crate::assert_array_relative_eq!(a.r() * b.r(), expected, 1E-10);
    }

    #[test]
    pub fn test_cmp_mul_assign() {
        let mut a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let b: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();
        let expected: DynArray<_, 1> = vec![1.5, 4.6, 12.0].into();

        a *= b;

        crate::assert_array_relative_eq!(a, expected, 1E-10);
    }

    #[test]
    pub fn test_scalar_mul() {
        let a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let expected: DynArray<_, 1> = vec![2.0, 4.0, 6.0].into();

        crate::assert_array_relative_eq!(2.0_f64 * a.r(), expected, 1E-10);
    }

    #[test]
    pub fn test_scalar_mul_assign() {
        let mut a: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let expected: DynArray<_, 1> = vec![2.0, 4.0, 6.0].into();

        a *= 2.0;

        crate::assert_array_relative_eq!(a.r(), expected, 1E-10);
    }

    #[test]
    pub fn test_cmp_div() {
        let a: DynArray<_, 1> = vec![1.5, 4.6, 12.0].into();
        let b: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let expected: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();

        crate::assert_array_relative_eq!(a.r() / b.r(), expected, 1E-10);
    }

    #[test]
    pub fn test_cmp_div_assign() {
        let mut a: DynArray<_, 1> = vec![1.5, 4.6, 12.0].into();
        let b: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();
        let expected: DynArray<_, 1> = vec![1.5, 2.3, 4.0].into();

        a /= b;

        crate::assert_array_relative_eq!(a, expected, 1E-10);
    }

    #[test]
    pub fn test_scalar_div() {
        let a: DynArray<_, 1> = vec![2.0, 4.0, 6.0].into();
        let expected: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();

        crate::assert_array_relative_eq!(a.r() / 2.0, expected, 1E-10);
    }

    #[test]
    pub fn test_scalar_div_assign() {
        let mut a: DynArray<_, 1> = vec![2.0, 4.0, 6.0].into();
        let expected: DynArray<_, 1> = vec![1.0, 2.0, 3.0].into();

        a /= 2.0;

        crate::assert_array_relative_eq!(a.r(), expected, 1E-10);
    }

    #[test]
    pub fn test_into_subview() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut a = DynArray::<f64, 3>::from_shape([7, 5, 9]);
        a.fill_from_equally_distributed(&mut rng);

        let view = a.r().into_subview([1, 0, 3], [2, 3, 4]);

        assert_eq!(view[[1, 2, 3]], a[[2, 2, 6]])
    }

    #[test]
    pub fn test_apply() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let alpha = 1.5;
        let beta = 3.6;

        let mut a = DynArray::<f64, 2>::from_shape([3, 2]);
        a.fill_from_equally_distributed(&mut rng);

        let mut x = DynArray::<f64, 1>::from_shape([2]);
        x.fill_from_equally_distributed(&mut rng);

        let mut y = DynArray::<f64, 1>::from_shape([3]);
        let y2 = y.eval();

        a.apply(alpha, &x, beta, &mut y);

        for (row_index, row) in a.row_iter().enumerate() {
            assert_relative_eq!(
                alpha * row.inner(&x).unwrap() + beta * y2[[row_index]],
                y[[row_index]],
                epsilon = 1E-10
            );
        }

        let mut x = DynArray::<f64, 2>::from_shape([2, 5]);
        x.fill_from_equally_distributed(&mut rng);

        let mut y = DynArray::<f64, 2>::from_shape([3, 5]);
        let y2 = y.eval();

        a.apply(alpha, &x, beta, &mut y);

        for (row_index, row) in a.row_iter().enumerate() {
            for (col_index, col) in x.col_iter().enumerate() {
                assert_relative_eq!(
                    alpha * row.inner(&col).unwrap() + beta * y2[[row_index, col_index]],
                    y[[row_index, col_index]],
                    epsilon = 1E-10
                );
            }
        }
    }

    #[test]
    pub fn test_dot() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut a = DynArray::<f64, 2>::from_shape([3, 2]);
        a.fill_from_equally_distributed(&mut rng);

        let mut x = DynArray::<f64, 1>::from_shape([2]);
        x.fill_from_equally_distributed(&mut rng);

        let mut y = DynArray::<f64, 1>::from_shape([3]);

        a.apply(1.0, &x, 0.0, &mut y);

        crate::assert_array_relative_eq!(y, a.dot(&x), 1E-10);

        let mut x = DynArray::<f64, 2>::from_shape([2, 5]);
        x.fill_from_equally_distributed(&mut rng);

        let mut y = DynArray::<f64, 2>::from_shape([3, 5]);

        a.apply(1.0, &x, 0.0, &mut y);

        crate::assert_array_relative_eq!(y, a.dot(&x), 1E-10);
    }
}
