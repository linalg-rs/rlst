//! Traits for properties of numbers.

/// Return the conjugate of an object.
pub trait Conj {
    /// Output type.
    type Output;
    /// Return the conjugate of an object.
    fn conj(&self) -> Self::Output;
}

/// Return the maximum of two objects.
pub trait Max<Other = Self> {
    /// Output type.
    type Output;
    /// Return the maximum of `self` and `other`.
    fn max(&self, other: &Other) -> Self::Output;
}

/// Return the minimum of two objects.
pub trait Min<Other = Self> {
    /// Output type.
    type Output;
    /// Return the maximum of `self` and `other`.
    fn min(&self, other: &Other) -> Self::Output;
}

/// Return the absolute value of an object.
pub trait Abs {
    /// Output type.
    type Output;

    /// Return the absolute value.
    fn abs(&self) -> Self::Output;
}

/// Return the square of an object.
pub trait Square {
    /// Output type
    type Output;

    /// Return the square of an object.
    fn square(&self) -> Self::Output;
}

/// Return the square of the absolute value.
pub trait AbsSquare {
    /// Output type
    type Output;

    /// Return the square of the absolute value.
    fn abs_square(&self) -> Self::Output;
}

/// Return the square root of the number
pub trait Sqrt {
    /// Output type
    type Output;

    /// Return the square root of the number.
    fn sqrt(&self) -> Self::Output;
}
