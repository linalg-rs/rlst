//! Linear spaces and their elements.

use super::Element;
use rlst_common::types::Scalar;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: Scalar;

    /// Type associated with elements of the space.
    type E<'b>: Element<Space = Self>
    where
        Self: 'b;

    /// Check if a space is the same as another space.
    fn is_same(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    /// Create a new vector from the space.
    fn zero(&self) -> Self::E<'_> {
        std::unimplemented!();
    }
}
