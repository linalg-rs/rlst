//! Traits for properties of numbers.

use std::cmp::Ordering;

/// Return the conjugate of an object.
pub trait Conj {
    /// Output type.
    type Output;
    /// Return the conjugate of an object.
    fn conj(self) -> Self::Output;
}

/// Return the maximum of two objects.
pub trait Max<Other = Self> {
    /// Output type.
    type Output;
    /// Return the maximum of `self` and `other`.
    fn max(self, other: Other) -> Self::Output;
}

/// Return the minimum of two objects.
pub trait Min<Other = Self> {
    /// Output type.
    type Output;
    /// Return the minimum of `self` and `other`.
    fn min(self, other: Other) -> Self::Output;
}

/// Return the comparison of two objects.
///
/// This is implemented as a separate trait since
/// Rust does not implement the [Ord] trait for
/// floating point types. For these it provides
/// a separate `total_cmp` function. But this
/// is not available for other types.
/// This trait makes up for this by providing a unified
/// interface for all types.
pub trait TotalCmp: Sized + Copy {
    /// Compare two elements
    fn total_cmp(self, other: Self) -> Ordering;

    /// self < other
    #[inline(always)]
    fn lt(self, other: Self) -> bool {
        matches!(self.total_cmp(other), Ordering::Less)
    }

    /// self <= other
    #[inline(always)]
    fn le(self, other: Self) -> bool {
        self.lt(other) || self.eq(other)
    }

    /// self >= other
    #[inline(always)]
    fn ge(self, other: Self) -> bool {
        self.gt(other) || self.eq(other)
    }

    /// self > other
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        matches!(self.total_cmp(other), Ordering::Greater)
    }

    /// self == other
    #[inline(always)]
    fn eq(self, other: Self) -> bool {
        matches!(self.total_cmp(other), Ordering::Equal)
    }
}

/// Return the absolute value of an object.
pub trait Abs {
    /// Output type.
    type Output;

    /// Return the absolute value.
    fn abs(self) -> Self::Output;
}

/// Return the square of an object.
pub trait Square {
    /// Output type
    type Output;

    /// Return the square of an object.
    fn square(self) -> Self::Output;
}

/// Return the square of the absolute value.
pub trait AbsSquare {
    /// Output type
    type Output;

    /// Return the square of the absolute value.
    fn abs_square(self) -> Self::Output;
}

/// Return the square root of the number
pub trait Sqrt {
    /// Output type
    type Output;

    /// Return the square root of the number.
    fn sqrt(self) -> Self::Output;
}

/// Return the reciprocal of the number.
pub trait Recip {
    /// Output type
    type Output;
    /// Return the reciprocal of the number.
    fn recip(self) -> Self::Output;
}

/// Return the exponential of the number.
pub trait Exp {
    /// Output type
    type Output;
    /// Return the exponential of the number.
    fn exp(self) -> Self::Output;
}

/// Return the natural logarithm of the number.
pub trait Ln {
    /// Output type
    type Output;
    /// Return the natural logarithm of the number.
    fn ln(self) -> Self::Output;
}

/// Return the sine of the number.
pub trait Sin {
    /// Output type
    type Output;
    /// Return the sine of the number.
    fn sin(self) -> Self::Output;
}

/// Return the cosine of the number.
pub trait Cos {
    /// Output type
    type Output;
    /// Return the cosine of the number.
    fn cos(self) -> Self::Output;
}

/// Return the tangent of the number.
pub trait Tan {
    /// Output type
    type Output;
    /// Return the tangent of the number.
    fn tan(self) -> Self::Output;
}

/// Return the inverse sine of the number.
pub trait Asin {
    /// Output type
    type Output;
    /// Return the inverse sine of the number.
    fn asin(self) -> Self::Output;
}

/// Return the inverse cosine of the number.
pub trait Acos {
    /// Output type
    type Output;
    /// Return the inverse cosine of the number.
    fn acos(self) -> Self::Output;
}

/// Return the inverse tangent of the number.
pub trait Atan {
    /// Output type
    type Output;
    /// Return the inverse tangent of the number.
    fn atan(self) -> Self::Output;
}

/// Return the hyperbolic sine of the number.
pub trait Sinh {
    /// Output type
    type Output;
    /// Return the hyperbolic sine of the number.
    fn sinh(self) -> Self::Output;
}

/// Return the hyperbolic cosine of the number.
pub trait Cosh {
    /// Output type
    type Output;
    /// Return the hyperbolic cosine of the number.
    fn cosh(self) -> Self::Output;
}

/// Return the hyperbolic tangent of the number.
pub trait Tanh {
    /// Output type
    type Output;
    /// Return the hyperbolic tangent of the number.
    fn tanh(self) -> Self::Output;
}

/// Return the inverse hyperbolic sine of the number.
pub trait Asinh {
    /// Output type
    type Output;
    /// Return the inverse hyperbolic sine of the number.
    fn asinh(self) -> Self::Output;
}

/// Return the inverse hyperbolic cosine of the number.
pub trait Acosh {
    /// Output type
    type Output;
    /// Return the inverse hyperbolic cosine of the number.
    fn acosh(self) -> Self::Output;
}

/// Return the inverse hyperbolic tangent of the number.
pub trait Atanh {
    /// Output type
    type Output;
    /// Return the inverse hyperbolic tangent of the number.
    fn atanh(self) -> Self::Output;
}
