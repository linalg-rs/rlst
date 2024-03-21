//! Normed spaces
use crate::ElementType;
use crate::InnerProductSpace;

use super::LinearSpace;
use rlst_dense::types::RlstScalar;

/// Normed space
pub trait NormedSpace: LinearSpace {
    /// Norm of a vector.
    fn norm(&self, x: &ElementType<Self>) -> <Self::F as RlstScalar>::Real;
}

impl<S: InnerProductSpace> NormedSpace for S {
    fn norm(&self, x: &ElementType<Self>) -> <Self::F as RlstScalar>::Real {
        let abs_square = self.inner(x, x).abs();
        abs_square.sqrt()
    }
}
