//! Traits for in-place operations.

use crate::types::Scalar;

/// Scale the object by a given factor.
pub trait ScaleInPlace {
    type T: Scalar;

    fn scale_in_place(&mut self, alpha: Self::T);
}

/// Fill the object from `other`.
pub trait FillFrom<Other> {
    fn fill_from(&mut self, other: &Other);
}

/// Add `alpha * other` to `self`.
pub trait SumInto<Other> {
    type T: Scalar;

    fn sum_into(&mut self, alpha: Self::T, other: &Other);
}

/// Set the diagonal of an object from a given iterator.
pub trait SetDiag {
    type T: Scalar;

    /// Set the diagonal from an iterator.
    ///
    /// The method sets the diagonal from the iterator up to the minimum of iterator
    /// length or number of diagonal elements.
    fn set_diag_from_iter<I: Iterator<Item = Self::T>>(&mut self, iter: I);

    /// Set the diagonal from a given slice.
    ///
    /// Produces an error if the length of the slice is not identical to
    /// the length of the diagonal.
    fn set_diag_from_slice(&mut self, diag: &[Self::T]);
}
