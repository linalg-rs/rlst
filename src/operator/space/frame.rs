//! A frame is a collection of elements of a space.

use crate::operator::ElementImpl;

use super::{Element, ElementContainer, ElementContainerMut, ElementType};

/// A frame is a collection of elements of a space.
pub trait Frame {
    /// Element type
    type E: ElementImpl;
    /// Iterator
    type Iter<'iter>: std::iter::Iterator<Item = &'iter ElementType<Self::E>>
    where
        Self: 'iter,
        Self::E: 'iter;
    /// Mutable iterator
    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut ElementType<Self::E>>
    where
        Self: 'iter,
        Self::E: 'iter;
    /// Get an element
    fn get(&self, index: usize) -> Option<&ElementType<Self::E>>;
    /// Get a mutable element
    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<Self::E>>;
    /// Modify an element
    //fn set(&mut self, axis_0: usize, axis_1: usize, val: <Self::E as ElementImpl>::F);
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
    fn push(&mut self, elem: Element<impl ElementContainer<E = Self::E>>);
    /// Evaluate
    fn evaluate(
        &self,
        coeffs: &[<Self::E as ElementImpl>::F],
        mut result: Element<impl ElementContainerMut<E = Self::E>>,
    ) {
        println!("coeffs: {:?}, {:?}", coeffs.len(), self.len());
        assert_eq!(coeffs.len(), self.len());
        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.axpy_inplace(coeff, elem.r());
        }
    }
}

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

    /*fn set(&mut self, axis_0: usize, axis_1: usize, val: <Self::E as ElementImpl>::F) {
        self.data.get_mut(axis_1).unwrap().set(axis_0, val);
    }*/

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
