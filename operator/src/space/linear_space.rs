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

    /// Clone an element of the space.
    fn clone<'a>(&'a self, elem: &Self::E<'a>) -> Self::E<'a> {
        let mut cloned = self.zero();
        cloned.sum_into(<Self::F as One>::one(), elem);
        cloned
    }
}

pub type ElementType<'a, Space> = <Space as LinearSpace>::E<'a>;
pub type FieldType<Space> = <Space as LinearSpace>::F;
