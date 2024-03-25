//! Inner product space

use super::LinearSpace;

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    fn inner(&self, x: &Self::E, other: &Self::E) -> Self::F;
}
