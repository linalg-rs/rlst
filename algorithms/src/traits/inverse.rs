//! Compute the inverse of an operator.
use rlst_common::types::RlstResult;

/// Compute the inverse.
pub trait Inverse {
    type Out;

    fn inverse(self) -> RlstResult<Self::Out>;
}
