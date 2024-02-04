//! Linear spaces and their elements.

use super::Element;
use num::One;
use rlst_common::types::Scalar;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F: Scalar;

    /// Type associated with elements of the space.
    type E: Element<F = Self::F>;

    /// Create a new vector from the space.
    fn zero(&self) -> Self::E;

    /// Create a new element from a given one.
    fn new_from(&self, elem: &Self::E) -> Self::E {
        let mut cloned = self.zero();
        cloned.axpy_inplace(<Self::F as One>::one(), elem);
        cloned
    }
}

pub type ElementType<Space> = <Space as LinearSpace>::E;
pub type FieldType<Space> = <Space as LinearSpace>::F;
