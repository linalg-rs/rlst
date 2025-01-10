//! Inner product space

use super::LinearSpace;

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    fn inner<'a>(&self, x: &Self::E<'a>, other: &Self::E<'a>) -> Self::F;
}
