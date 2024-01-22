//! Linear spaces and their elements.

use super::Element;
use rlst_common::types::RlstScalar;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: RlstScalar;

    /// Type associated with elements of the space.
    type E<'b>: Element<Space = Self>
    where
        Self: 'b;

    /// Create a new vector from the space.
    fn create_element(&self) -> Self::E<'_> {
        std::unimplemented!();
    }
}
