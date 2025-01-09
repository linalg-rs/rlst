//! Linear spaces and their elements.

use super::Element;
use crate::dense::types::RlstScalar;
use num::One;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: RlstScalar;

    /// Type associated with elements of the space.
    type E<'a>: Element<'a, F = Self::F>
    where
        Self: 'a;

    /// Create a new vector from the space.
    fn zero(&self) -> Self::E<'_>;
}
/// Element type
pub type ElementType<'a, Space> = <Space as LinearSpace>::E<'a>;
/// Field type
pub type FieldType<Space> = <Space as LinearSpace>::F;
