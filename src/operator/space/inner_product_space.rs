//! Inner product space

use super::LinearSpace;

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    fn inner<'b>(&self, x: &Self::E<'b>, other: &Self::E<'b>) -> Self::F
    where
        Self: 'b;
}
