//! A frame is a collection of elements of a space.

use crate::Element;

/// A frame
///
/// A frame is a collection of elements of a space.
pub trait Frame {
    /// Element type
    type E: Element;
    /// Iterator
    type Iter<'iter>: std::iter::Iterator<Item = &'iter Self::E>
    where
        Self: 'iter;
    /// Mutable iterator
    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut Self::E>
    where
        Self: 'iter;
    /// Get an element
    fn get(&self, index: usize) -> Option<&Self::E>;
    /// Get a mutable element
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::E>;
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
    fn push(&mut self, elem: Self::E);
    /// Evaluate
    fn evaluate(&self, coeffs: &[<Self::E as Element>::F], result: &mut Self::E) {
        assert_eq!(coeffs.len(), self.len());
        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.axpy_inplace(coeff, elem);
        }
    }
}

/// A vector frame
pub struct VectorFrame<Elem: Element> {
    data: Vec<Elem>,
}

impl<Elem: Element> VectorFrame<Elem> {
    /// Create a new vector frame
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl<Elem: Element> Default for VectorFrame<Elem> {
    fn default() -> Self {
        Self::new()
    }
}

impl<Elem: Element> Frame for VectorFrame<Elem> {
    type E = Elem;

    type Iter<'iter> = std::slice::Iter<'iter, Self::E>
    where
        Self: 'iter;

    type IterMut<'iter> = std::slice::IterMut<'iter, Self::E>
    where
        Self: 'iter;

    fn get(&self, index: usize) -> Option<&Self::E> {
        self.data.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Self::E> {
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

    fn push(&mut self, elem: Self::E) {
        self.data.push(elem)
    }
}
