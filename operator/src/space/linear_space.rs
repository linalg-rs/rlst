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
    type E<'elem>: Element<'elem, Space = Self>
    where
        Self: 'elem;

    /// Check if a space is the same as another space.
    fn is_same(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }

    /// Create a new vector from the space.
    fn zero(&self) -> Self::E<'_>;

    /// Clone an element of the space.
    fn clone<'space, 'elem>(&'space self, elem: &Self::E<'elem>) -> Self::E<'space>
    where
        'space: 'elem,
    {
        let mut cloned = self.zero();
        cloned.sum_into(<Self::F as One>::one(), elem);

        // Manually ensure that the new variable as the lifetime of Space
        unsafe { std::mem::transmute(cloned) }
    }
}

pub type ElementType<'elem, Space> = <Space as LinearSpace>::E<'elem>;
pub type FieldType<Space> = <Space as LinearSpace>::F;
