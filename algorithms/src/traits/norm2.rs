//! The 2-norm of an object.
use rlst_common::types::Scalar;

/// Compute the 2-norm of an object.
pub trait Norm2 {
    type T: Scalar;

    fn norm2(&self) -> <Self::T as Scalar>::Real;
}
