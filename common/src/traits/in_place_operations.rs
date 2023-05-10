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
