//! Linear spaces and their elements.

use super::Element;
use crate::Scalar;

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

    /// Create a new vector from the space.
    fn create_element<'b>(&'b self) -> Self::E<'b> {
        std::unimplemented!();
    }
}
