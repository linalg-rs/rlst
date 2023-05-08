//! The 2-norm of an object.
use rlst_common::types::{RlstResult, Scalar};

/// Compute the 2-norm of an object.
pub trait Norm2 {
    type T: Scalar;

    fn norm2(self) -> RlstResult<<Self::T as Scalar>::Real>;
}
