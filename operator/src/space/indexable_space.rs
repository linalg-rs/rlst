//! Indexable space
use super::LinearSpace;

/// Indexable space
pub trait IndexableSpace: LinearSpace {
    /// Dimension
    fn dimension(&self) -> usize;
}
