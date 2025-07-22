//! Abstract linear spaces and their elements.

use std::rc::Rc;

use crate::operator::element::{Element, ElementContainer};

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace {
    /// Field Type.
    type F;

    /// Concrete type associated with elements of the space.
    type E;

    /// Create a new zero element from the space.
    fn zero(space: Rc<Self>) -> Element<ElementContainer<Self::E>>;
}

/// Defne and return the space of an element.
pub trait ElementSpace {
    ///  Space type.
    type Space;

    /// Get the space of the element.
    fn space(&self) -> &Self::Space;
}

/// A dual space
pub trait DualSpace<ContainerX, ContainerY>: LinearSpace {
    /// Space type
    type Space: LinearSpace<F = Self::F>;

    /// Dual pairing
    fn dual_pairing(&self, x: &Element<ContainerX>, y: &Element<ContainerY>) -> Self::F;
}

/// Indexable space
pub trait IndexableSpace: LinearSpace {
    /// Dimension
    fn dimension(&self) -> usize;
}

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    fn inner_product(&self, x: &Self::E, other: &Self::E) -> Self::F;
}

/// Normed space
pub trait NormedSpace<Container>: LinearSpace {
    /// The output type of the norm.
    type Output;

    /// Norm of a vector.
    fn norm(&self, x: &Element<Container>) -> Self::Output;
}

/// A frame is a collection of elements of a space.
pub trait Frame {
    ///  The underlying linear space.
    type Space: LinearSpace;
    /// Iterator
    type Iter<'iter>: std::iter::Iterator<
        Item = &'iter Element<ElementContainer<<Self::Space as LinearSpace>::E>>,
    >
    where
        Self: 'iter;
    /// Mutable iterator
    type IterMut<'iter>: std::iter::Iterator<
        Item = &'iter mut Element<ElementContainer<<Self::Space as LinearSpace>::E>>,
    >
    where
        Self: 'iter;
    /// Get a reference to an element
    fn get(
        &self,
        index: usize,
    ) -> Option<&Element<ElementContainer<<Self::Space as LinearSpace>::E>>>;
    /// Get a mutable reference to an element
    fn get_mut(
        &mut self,
        index: usize,
    ) -> Option<&mut Element<ElementContainer<<Self::Space as LinearSpace>::E>>>;
    /// Number of elements
    fn len(&self) -> usize;
    /// Is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get iterator
    fn iter(&self) -> Self::Iter<'_>;
    /// Get mutable iterator
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    /// Add an element
    fn push(&mut self, elem: Element<ElementContainer<<Self::Space as LinearSpace>::E>>);

    /// Remove the last element and return it. If the frame is empty, return None.
    fn pop(&mut self) -> Option<Element<ElementContainer<<Self::Space as LinearSpace>::E>>>;
}
