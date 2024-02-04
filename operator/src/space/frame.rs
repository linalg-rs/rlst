//! A frame is a collection of elements of a space.

use crate::Element;

pub trait Frame {
    type E: Element;

    type Iter<'iter>: std::iter::Iterator<Item = &'iter Self::E>
    where
        Self: 'iter;

    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut Self::E>
    where
        Self: 'iter;

    fn get(&self, index: usize) -> Option<&Self::E>;

    fn get_mut(&mut self, index: usize) -> Option<&mut Self::E>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> Self::Iter<'_>;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn push(&mut self, elem: Self::E);

    fn evaluate(&self, coeffs: &[<Self::E as Element>::F], result: &mut Self::E) {
        assert_eq!(coeffs.len(), self.len());
        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.axpy_inplace(coeff, elem);
        }
    }
}

pub struct VectorFrame<Elem: Element> {
    data: Vec<Elem>,
}

impl<Elem: Element> VectorFrame<Elem> {
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
