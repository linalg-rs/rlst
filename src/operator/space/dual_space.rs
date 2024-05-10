//! Dual space
use super::LinearSpace;
use crate::InnerProductSpace;

/// A dual space
pub trait DualSpace<Other: LinearSpace<F = Self::F>>: LinearSpace {
    /// Dual pairing
    fn dual_pairing(&self, x: &Self::E, other: &Other::E) -> Self::F;
}

impl<S: InnerProductSpace> DualSpace<S> for S {
    fn dual_pairing(&self, x: &Self::E, other: &S::E) -> Self::F {
        self.inner(x, other)
    }
}
