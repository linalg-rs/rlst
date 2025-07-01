//! Traits for array properties and operations.

use paste::paste;
use std::ops::MulAssign;

use num::{One, Zero};

use crate::base_types::MemoryLayout;

use super::iterators::{ArrayIterator, ArrayIteratorMut, GetDiagMut};

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
    fn is_contiguous(&self) -> bool
    where
        Self: Shape<NDIM>,
    {
        self.memory_layout() != MemoryLayout::Unknown
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

/// Convert to a new type.
///
/// This trait is used to convert an array of one type into an array of another type.
/// It depends on the `Into` trait to convert each element of the array.
pub trait ToType {
    /// The element type of the array.
    type Item;
    /// The output type of the array.
    type Output<T>;

    /// Convert the array into a new array.
    fn to_type<T>(self) -> Self::Output<T>
    where
        Self::Item: Into<T>;
}

macro_rules! unary_op_trait {
    ($name:ident, $op:ident) => {
        paste! {
            #[doc = "Apply the unary operation `" [<$op>] "` to each element of the array."]
            pub trait [<ArrayOp $name>] {
                /// The output array type.
                type Output;

                /// Apply the unary operation `$op` to each element of the array.
                fn $op(self) -> Self::Output;
            }

        }
    };
}

unary_op_trait!(Abs, abs);
unary_op_trait!(AbsSquare, abs_square);
unary_op_trait!(Square, square);
unary_op_trait!(Sqrt, sqrt);
unary_op_trait!(Exp, exp);
unary_op_trait!(Ln, ln);
unary_op_trait!(Recip, recip);
unary_op_trait!(Sin, sin);
unary_op_trait!(Cos, cos);
unary_op_trait!(Tan, tan);
unary_op_trait!(Asin, asin);
unary_op_trait!(Acos, acos);
unary_op_trait!(Atan, atan);
unary_op_trait!(Sinh, sinh);
unary_op_trait!(Cosh, cosh);
unary_op_trait!(Tanh, tanh);
unary_op_trait!(Asinh, asinh);
unary_op_trait!(Acosh, acosh);
unary_op_trait!(Atanh, atanh);
