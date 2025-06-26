//! Dense matrix traits

pub mod accessors;
pub mod number_traits;

use std::ops::MulAssign;

pub use accessors::*;
use num::{One, Zero};

use crate::dense::types::TransMode;

pub use super::types::RlstScalar;
use super::{
    array::{SliceArray, SliceArrayMut},
    types::RlstResult,
};
pub use number_traits::*;

/// Memory layout of an object
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Column major
    ColumnMajor,
    /// Row major
    RowMajor,
    /// Unknown
    Unknown,
}

///Base item type of an array.
pub trait BaseItem {
    /// Item type
    type Item;
}

/// Shape of an object
pub trait Shape<const NDIM: usize> {
    /// Return the shape of the object.
    fn shape(&self) -> [usize; NDIM];
}

/// Stride of an object
pub trait Stride<const NDIM: usize> {
    /// Return the stride of the object.
    fn stride(&self) -> [usize; NDIM];

    /// Return the memory layout
    fn memory_layout(&self) -> MemoryLayout {
        MemoryLayout::Unknown
    }
}

/// Number of elements
pub trait NumberOfElements {
    /// Return the number of elements.
    fn number_of_elements(&self) -> usize;
}

/// Resize in place
pub trait ResizeInPlace<const NDIM: usize> {
    /// Resize an operator in place
    fn resize_in_place(&mut self, shape: [usize; NDIM]);
}

/// Multiply into
pub trait MultInto<First, Second>: BaseItem {
    /// Multiply First * Second and sum into Self
    fn simple_mult_into(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
    {
        self.mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into
    fn mult_into(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

/// Gemm
pub trait Gemm: Sized {
    /// Gemm
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        transa: TransMode,
        transb: TransMode,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        rsa: usize,
        csa: usize,
        b: &[Self],
        rsb: usize,
        csb: usize,
        beta: Self,
        c: &mut [Self],
        rsc: usize,
        csc: usize,
    );
}

/// Multiply into with resize
pub trait MultIntoResize<First, Second>: BaseItem {
    /// Multiply First * Second and sum into Self. Allow to resize Self if necessary
    fn simple_mult_into_resize(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
    {
        self.mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into with resize
    fn mult_into_resize(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

// /// Default iterator.
// pub trait DefaultIterator {
//     /// Item type
//     type Item;
//     /// Iterator type
//     type Iter<'a>: std::iter::Iterator<Item = Self::Item>
//     where
//         Self: 'a;
//     /// Get iterator
//     fn iter(&self) -> Self::Iter<'_>;
// }

// /// Mutable default iterator.
// pub trait DefaultIteratorMut {
//     /// Item type
//     type Item;
//     /// Iterator
//     type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
//     where
//         Self: 'a;
//     /// Get mutable iterator
//     fn iter_mut(&mut self) -> Self::IterMut<'_>;
// }

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator: BaseItem {
    /// Iterator
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::Item)>
    where
        Self: 'a;
    /// Get iterator
    fn iter_aij(&self) -> Self::Iter<'_>;
}

/// Helper trait that returns from an enumeration iterator a new iterator
/// that converts the 1d index into a multi-index.
pub trait AsMultiIndex<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> {
    /// Get multi-index
    fn multi_index(
        self,
        shape: [usize; NDIM],
    ) -> crate::dense::array::iterators::MultiIndexIterator<I, NDIM>;
}

/// A helper trait to implement generic operators over matrices.
pub trait AsOperatorApply: BaseItem {
    /// Apply the operator to a vector.
    fn apply_extended(
        &self,
        alpha: Self::Item,
        x: &[Self::Item],
        beta: Self::Item,
        y: &mut [Self::Item],
    );
}

/// Provides a default iterator over elements of an array.
///
/// The returned iterator is expected to iterate through the array
/// in column major order independent of the underlying memory layout.
pub trait ArrayIterator: BaseItem {
    /// Type of the iterator.
    type Iter<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces elements of type `Self::Item`.
    fn iter(&self) -> Self::Iter<'_>;
}

/// Provides a default mutable iterator over elements of an array.
///
/// The returned iterator is expected to iterate through the array
/// in column major order independent of the underlying memory layout.
pub trait ArrayIteratorMut: BaseItem {
    /// Type of the iterator.
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces mutable references to elements of type `Self::Item`.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

/// Get an iterator to the diagonal of an array.
pub trait GetDiag: BaseItem {
    /// Type of the iterator.
    type Iter<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    /// Return an iterator for the diagonal of an array.
    fn diag_iter(&self) -> Self::Iter<'_>;
}

/// Get a mutable iterator to the diagonal of an array.
pub trait GetDiagMut: BaseItem {
    /// Tyepof the iterator.
    type Iter<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Return a mutable iterator for the diagonal of an array.
    fn diag_iter_mut(&mut self) -> Self::Iter<'_>;
}

/// Fill an array with values from another array.
pub trait FillFrom<Other> {
    /// Fill an array with values from another array.
    fn fill_from(&mut self, other: &Other);
}

/// Fill an array with values from another array and allow resizing.
pub trait FillFromResize<Other> {
    /// Fill an array with values from another array and allow resizing.
    fn fill_from_resize(&mut self, other: &Other);
}

/// Sum into an array with values from another array.
pub trait SumFrom<Other> {
    /// Sum into array with values from `other`.
    fn sum_from(&mut self, other: &Other);
}

/// Componentwise Multiply other array into this array.
pub trait CmpMulFrom<Other> {
    /// Multiply other array into this array.
    fn cmp_mult_from(&mut self, other: &Other);
}

/// Componentwise form `Self = Self * Other1 + Other2`.
pub trait CmpMulAddFrom<Other1, Other2> {
    /// Componentwise form `Self = Self * Other1 + Other2`.
    fn cmp_mul_add_from(&mut self, other1: &Other1, other2: &Other2);
}

/// Fill an array with a specific value.
pub trait FillWithValue: BaseItem {
    /// Fill an array with a specific value.
    fn fill_with_value(&mut self, value: Self::Item);
}

/// Fill an array with zero values.
pub trait SetZero {
    /// Fill an array with zero values.
    fn set_zero(&mut self);
}

/// Fill an array with one values.
pub trait SetOne {
    /// Fill an array with one values.
    fn set_one(&mut self);
}

impl<T> SetZero for T
where
    T: FillWithValue,
    T::Item: Zero,
{
    fn set_zero(&mut self) {
        self.fill_with_value(Zero::zero());
    }
}

impl<T> SetOne for T
where
    T: FillWithValue,
    T::Item: One,
{
    fn set_one(&mut self) {
        self.fill_with_value(One::one());
    }
}

/// Set the diagonal of an array to one and the off-diagonals to zero.
pub trait SetIdentity {
    /// Fill the diagonal of an array with the value 1 and all other elements zero.
    fn set_identity(&mut self);
}

impl<Item, T> SetIdentity for T
where
    T: FillWithValue<Item = Item> + GetDiagMut<Item = Item>,
    Item: Zero + One,
{
    fn set_identity(&mut self) {
        self.set_zero();
        self.diag_iter_mut().for_each(|elem| *elem = One::one());
    }
}

/// Scale all elements by a value `alpha`.
pub trait ScaleInPlace: BaseItem {
    /// Scale all elements by a value `alpha`.
    fn scale_in_place(&mut self, alpha: Self::Item);
}

impl<T> ScaleInPlace for T
where
    T: ArrayIteratorMut,
    T::Item: MulAssign<T::Item> + Copy,
{
    fn scale_in_place(&mut self, alpha: Self::Item) {
        self.iter_mut().for_each(|elem| *elem *= alpha);
    }
}

/// Compute the trace of an operator.
pub trait Trace: BaseItem {
    /// Compute the trace.
    fn trace(&self) -> Self::Item;
}

/// Sum all elements of an array.
pub trait Sum: BaseItem {
    /// Compute the sum of all elemenets.
    fn sum(&self) -> Self::Item;
}

impl<T> Sum for T
where
    T: ArrayIterator,
    T::Item: std::iter::Sum,
{
    fn sum(&self) -> Self::Item {
        self.iter().sum()
    }
}

/// Compute the length of an array.
///
/// For multi-dimensional array the length is the product of the dimensions.
pub trait Len {
    /// Return the length of the array.
    fn len(&self) -> usize;

    /// Return true if the array has no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compute the inner product with another vector.
pub trait Inner<Other = Self> {
    /// The Item type of the inner product.
    type Output;
    /// Return the inner product of `Self` with `Other`.
    fn inner(&self, other: &Other) -> Self::Output;
}

/// Return the supremum norm of an array.
pub trait NormSup {
    /// The Item type of the norm.
    type Output;

    /// Return the supremum norm.
    fn norm_sup(&self) -> Self::Output;
}

/// Return the 1-norm of an array.
pub trait NormOne {
    /// The Item type of the norm.
    type Output;

    /// Return the 1-norm.
    fn norm_1(&self) -> Self::Output;
}

/// Return the 2-norm of an array.
pub trait NormTwo {
    /// The Item type of the norm.
    type Output;

    /// Return the 2-norm.
    fn norm_2(&self) -> Self::Output;
}

/// Return the array of conjugate numbers.
pub trait ConjArray {
    /// The output type of the conjugate array.
    type Output;

    /// Return the conjugate array.
    fn conj(self) -> Self::Output;
}

/// Evaluate array into a new array.
pub trait EvaluateArray {
    /// The output type of the evaluated array.
    type Output;

    /// Evaluate the array into a new array.
    fn eval(&self) -> Self::Output;
}

/// Extend Rust into to all elements of an array.
pub trait IntoArray {
    /// The element type of the array.
    type Item;
    /// The output type of the array.
    type Output<T>;

    /// Convert the array into a new array.
    fn into_array<T>(self) -> Self::Output<T>
    where
        Self::Item: Into<T>;
}

/// Return a reference to the underlying data of an array as a 2-dimensional array.
///
///This is useful to get a guaranted 2d view onto a matrix or vector for functions that require
///a 2d array.
pub trait As2dArray<const NDIM: usize> {
    type ArrayImpl: BaseItem<Item = Self::Item>
        + Shape<NDIM>
        + RawAccess<Item = Self::Item>
        + Stride<NDIM>;

    type Item;

    /// Try to return a reference to the underlying data as a 2-dimensional column-major array.
    fn as_2d_col_major_array(&self) -> RlstResult<SliceArray<'_, Self::Item, 2>>;
}

/// Return a reference to the underlying data of an array as a 2-dimensional mutable array.
///
///This is useful to get a guaranted 2d view onto a matrix or vector for functions that require
///a 2d array.
pub trait As2dArrayMut<const NDIM: usize> {
    type ArrayImpl: BaseItem<Item = Self::Item>
        + Shape<NDIM>
        + RawAccessMut<Item = Self::Item>
        + Stride<NDIM>;

    type Item;

    /// Try to return a reference to the underlying data as a 2-dimensional column-major array.
    fn as_2d_col_major_array_mut(&mut self) -> RlstResult<SliceArrayMut<'_, Self::Item, 2>>;
}
