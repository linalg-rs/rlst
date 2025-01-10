//! A frame is a collection of elements of a space.

use std::marker::PhantomData;

use crate::operator::Element;

/// A frame is a collection of elements of a space.
pub trait Frame<'a> {
    /// Element type
    type E: Element<'a>;
    /// Iterator
    type Iter<'iter>: std::iter::Iterator<Item = &'iter Self::E>
    where
        Self: 'iter,
        Self::E: 'iter;
    /// Mutable iterator
    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut Self::E>
    where
        Self: 'iter,
        Self::E: 'iter;
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
    fn evaluate(&self, coeffs: &[<Self::E as Element<'a>>::F], result: &mut Self::E) {
        assert_eq!(coeffs.len(), self.len());
        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.axpy_inplace(coeff, elem);
        }
    }
}

/// A vector frame
pub struct VectorFrame<'a, Elem: Element<'a>> {
    data: Vec<Elem>,
    _marker: PhantomData<&'a ()>,
}

impl<'a, Elem: Element<'a>> VectorFrame<'a, Elem> {
    /// Create a new vector frame
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            _marker: PhantomData,
        }
    }
}

impl<'a, Elem: Element<'a>> Default for VectorFrame<'a, Elem> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, Elem: Element<'a>> Frame<'a> for VectorFrame<'a, Elem> {
    type E = Elem;

    type Iter<'iter>
        = std::slice::Iter<'iter, Self::E>
    where
        Self: 'iter;

    type IterMut<'iter>
        = std::slice::IterMut<'iter, Self::E>
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
