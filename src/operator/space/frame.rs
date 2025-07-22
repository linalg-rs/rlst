//! A frame is a collection of elements of a space.

use crate::operator::ElementImpl;

use super::{Element, ElementContainer, ElementContainerMut, ElementType};

/// A vector frame
pub struct VectorFrame<Elem: ElementImpl> {
    data: Vec<ElementType<Elem>>,
}

impl<Elem: ElementImpl> VectorFrame<Elem> {
    /// Create a new vector frame
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl<Elem: ElementImpl> Default for VectorFrame<Elem> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Elem: ElementImpl> Frame for VectorFrame<Elem> {
    type E = Elem;

    type Iter<'iter>
        = std::slice::Iter<'iter, ElementType<Self::E>>
    where
        Self: 'iter;

    type IterMut<'iter>
        = std::slice::IterMut<'iter, ElementType<Self::E>>
    where
        Self: 'iter;

    fn get(&self, index: usize) -> Option<&ElementType<Self::E>> {
        self.data.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<Self::E>> {
        self.data.get_mut(index)
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn iter(&self) -> Self::Iter<'_> {
        self.data.iter()
    }

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        self.data.iter_mut()
    }

    fn push(&mut self, elem: Element<impl ElementContainer<E = Self::E>>) {
        self.data.push(elem.duplicate())
    }
}
