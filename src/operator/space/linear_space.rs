//! Linear spaces and their elements.

use std::rc::Rc;

use super::{ConcreteElementContainer, Element, ElementImpl};
use crate::dense::types::RlstScalar;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: RlstScalar;

    /// Type associated with elements of the space.
    type E: ElementImpl<F = Self::F>;

    /// Create a new zero element from the space.
    fn zero(space: Rc<Self>) -> Element<ConcreteElementContainer<Self::E>>;
}
/// Element type
pub type ElementImplType<Space> = <Space as LinearSpace>::E;
/// Field type
pub type FieldType<Space> = <Space as LinearSpace>::F;

/// Create a new zero element from a given space.
pub fn zero_element<Space: LinearSpace>(
    space: Rc<Space>,
) -> Element<ConcreteElementContainer<ElementImplType<Space>>> {
    Space::zero(space)
}
