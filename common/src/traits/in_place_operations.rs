//! Traits for in-place operations.

use crate::types::Scalar;

/// Scale the object by a given factor.
pub trait Scale {
    type T: Scalar;

    fn scale(&mut self, alpha: Self::T);
}

/// Fill the object from `other`.
pub trait FillFrom<Other> {
    fn fill_from(&mut self, other: &Other);
}

/// Add `alpha * x` to `self`.
pub trait SumInto {
    type T: Scalar;

    fn mult_sum_into(&self, alpha: Self::T, x: &Self);
}
