//! Basic traits for elements of vector spaces.

use crate::{Array, base_types::MemoryLayout};

use super::UnsafeRandom1DAccessByValue;

/// Define a basic item type associated with an object.
pub trait BaseItem {
    /// Item type.
    type Item: Copy + Default;
}

///  Define a type that holds a reference to the current object.
///
/// This is useful to avoid double definition of operations for
/// references and owned types.
pub trait AsOwnedRefType {
    /// The reference type of the array.
    type RefType<'a>
    where
        Self: 'a;

    /// Return the reference type.
    fn r(&self) -> Self::RefType<'_>;
}

/// Define a type that holds a mutable reference to the current object.
pub trait AsOwnedRefTypeMut {
    /// The mutable reference type of the array.
    type RefTypeMut<'a>
    where
        Self: 'a;

    /// Return the mutable reference type.
    fn r_mut(&mut self) -> Self::RefTypeMut<'_>;
}

/// Associate a shape with a given object.
pub trait Shape<const NDIM: usize> {
    /// Return the shape of the object.
    fn shape(&self) -> [usize; NDIM];

    /// Return the length, which for n dimension is the product of the dimensions.
    #[inline(always)]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Return true if the array is empty.
    ///
    /// An array is empty if at least one dimension is zero.
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Associate a strice with a given object.
///
/// A stride describes the memory layout of an array.
/// If the stride is e.g. `[1, 5]`, this means
/// moving from one to the next element in the first
/// dimension jumps one memory entry and in the second
/// dimension jumps 5 memory entries.
pub trait Stride<const NDIM: usize> {
    /// Return the stride of the object.
    fn stride(&self) -> [usize; NDIM];

    /// Return the memory layout.
    ///
    /// A column-major layout has entries continuous in memory
    /// starting with the first outer dimension. A row-major layout
    /// has entries continuous in memory starting with the last outer dimension.
    ///
    /// Possible return values are [MemoryLayout::ColumnMajor], [MemoryLayout::RowMajor],
    /// and [MemoryLayout::Unknown].
    fn memory_layout(&self) -> MemoryLayout
    where
        Self: Shape<NDIM>,
    {
        let stride = self.stride();
        let shape = self.shape();
        let mut factor = 1;
        if stride[0] == 1 {
            // Contiguous 1d arrays are always column-major.
            if NDIM == 1 {
                return MemoryLayout::ColumnMajor;
            }
            // Check if strides are column-major
            for i in 1..NDIM {
                factor *= shape[i - 1];
                if stride[i] != factor {
                    return MemoryLayout::Unknown;
                }
            }
            MemoryLayout::ColumnMajor
        } else if stride[NDIM - 1] == 1 {
            // Check if strides are row-major
            for i in 2..NDIM + 1 {
                factor *= shape[NDIM - i + 1];
                if stride[NDIM - i] != factor {
                    return MemoryLayout::Unknown;
                }
            }
            MemoryLayout::RowMajor
        } else {
            MemoryLayout::Unknown
        }
    }

    /// Return true if the array is contiguous in memory.
    ///
    /// This is the case of the data is either row-major or column-major.
    /// Contgiuous memory orderings with respect to other layouts are not supported.
    fn is_contiguous(&self) -> bool
    where
        Self: Shape<NDIM>,
    {
        self.memory_layout() != MemoryLayout::Unknown
    }
}

/// Number of elements.
pub trait NumberOfElements {
    /// Return the number of elements.
    fn number_of_elements(&self) -> usize;
}

/// Resize in place.
pub trait ResizeInPlace<const NDIM: usize> {
    /// Resize an object in place.
    fn resize_in_place(&mut self, shape: [usize; NDIM]);
}

/// Fill an object with values from another object.
///
/// Behaviour if `self` and `other` have incompatible sizes
/// is not determined.
pub trait FillFrom<Other> {
    /// Fill `self` with values from `other`.
    fn fill_from(&mut self, other: &Other);
}

/// Fill an array with values from an iterator.
///
/// Behaviour if `self` and `iter` have incompatible sizes
/// is not determined.
pub trait FillFromIter<Iter: Iterator> {
    /// Fill `self` with values from `iter`.
    fn fill_from_iter(&mut self, iter: Iter);
}

/// Fill from another object and resize if necessary.
pub trait FillFromResize<Other> {
    /// Fill `self` with values from `other` and resize if necessary.
    fn fill_from_resize(&mut self, other: &Other);
}

/// Sum into current object from another object.
pub trait SumFrom<Other> {
    /// Sum values from `other` into `self`.
    fn sum_from(&mut self, other: &Other);
}

/// Componentwise Multiply other array into this array.
pub trait CmpMulFrom<Other> {
    /// Componentwise multiply the elements of `self` with `other` and store in `self`.
    fn cmp_mul_from(&mut self, other: &Other);
}

/// Multiply with a scalar.
pub trait ScalarMul<Scalar> {
    /// Output of multiplication with a scalar.
    type Output;
    /// Multiply `self` with `scalar`.
    fn scalar_mul(self, scalar: Scalar) -> Self::Output;
}

/// Componentwise form `Self = Self * Other1 + Other2`.
pub trait CmpMulAddFrom<Other1, Other2> {
    /// Componentwise form `Self = Self * Other1 + Other2`.
    fn cmp_mul_add_from(&mut self, other1: &Other1, other2: &Other2);
}

/// Fill an object with a specific value.
pub trait FillWithValue: BaseItem {
    /// Fill `self` with `value`.
    fn fill_with_value(&mut self, value: Self::Item);
}

/// Set elements current object to zero.
pub trait SetZero {
    /// Set all elements of `self` to `zero`.
    fn set_zero(&mut self);
}

/// Set elements of current object to one.
pub trait SetOne {
    /// Set all elemnets of `self` to `zero`.
    fn set_one(&mut self);
}

/// Set the current object to the identity element.
pub trait SetIdentity {
    /// Set `self` to be the identity.
    fn set_identity(&mut self);
}

/// Scale in place.
pub trait ScaleInPlace: BaseItem {
    /// Scale `self` by `alpha`.
    fn scale_inplace(&mut self, alpha: Self::Item);
}

/// Compute the trace of an object.
pub trait Trace: BaseItem {
    /// Return the trace of `self`.
    fn trace(&self) -> Self::Item;
}

/// Sum all elements.
pub trait Sum: BaseItem {
    /// Compute the sum of all elements of `self`.
    fn sum(&self) -> Self::Item;
}

/// Compute the length of an object.
///
/// For multi-dimensional array the length is the product of the dimensions.
pub trait Len {
    /// Return the length of `self`.
    fn len(&self) -> usize;

    /// Return true if `self` is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Return the conjugate object.
///
/// For an array this is the array of conjugate entries.
pub trait ConjObject {
    /// The output type.
    type Output;

    /// Return the conjugation of `self`.
    fn conj(self) -> Self::Output;
}

/// Evaluate into a new object.
pub trait EvaluateObject {
    /// The output type of the evaluated object.
    type Output;

    /// Evaluate the object.
    fn eval(&self) -> Self::Output;
}

/// Evaluate array into a new row-major array.
pub trait EvaluateRowMajorArray {
    /// The output type of the evaluated array.
    type Output;

    /// Evaluate the array into a new row-major array.
    fn eval_row_major(&self) -> Self::Output;
}

/// Dispatch the evaluation of an array to an actual implementation.
pub trait DispatchEval<const NDIM: usize> {
    /// The output type.
    type Output;

    /// The implementation type.
    type ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue;

    /// Dispatch the evaluation of the array to an actual implementation.
    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output;
}

/// Dispatch the row-major evaluation of an array to an actual implementation.
pub trait DispatchEvalRowMajor<const NDIM: usize> {
    /// The output type of the evaluated array.
    type Output;

    /// The implementation type of the array.
    type ArrayImpl: Shape<NDIM> + UnsafeRandom1DAccessByValue;

    /// Dispatch the evaluation of the array to an actual implementation.
    fn dispatch(&self, arr: &Array<Self::ArrayImpl, NDIM>) -> Self::Output;
}

/// Convert to a new type.
///
/// This trait is used to convert an object of one type into an object of another type.
pub trait ToType<T> {
    /// The element type of the array.
    type Item;
    /// The output type of the array.
    type Output;

    /// Convert the array into a new array.
    fn into_type(self) -> Self::Output
    where
        Self::Item: Into<T>;
}
