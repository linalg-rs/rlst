//! Linear spaces and their elements.

use std::rc::Rc;

use super::Element;
use crate::dense::types::RlstScalar;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: RlstScalar;

    /// Type associated with elements of the space.
    type E: Element<F = Self::F>;

    /// Create a new vector from the space.
    fn zero(space: Rc<Self>) -> Self::E;
}
/// Element type
pub type ElementType<Space> = <Space as LinearSpace>::E;
/// Field type
pub type FieldType<Space> = <Space as LinearSpace>::F;
