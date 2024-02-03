//! A frame is a collection of elements of a space.

use crate::{Element, ElementType, FieldType, LinearSpace};

pub trait Frame<'space: 'elem, 'elem> {
    type Space: LinearSpace + 'space;

    type Iter<'iter>: std::iter::Iterator<Item = &'iter ElementType<'elem, Self::Space>>
    where
        Self: 'elem,
        'elem: 'iter;

    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut ElementType<'elem, Self::Space>>
    where
        Self: 'elem,
        'elem: 'iter;

    fn get(&self, index: usize) -> Option<&ElementType<'elem, Self::Space>>;

    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<'elem, Self::Space>>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> Self::Iter<'_>;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn space(&self) -> &Self::Space;

    fn push(&mut self, elem: <Self::Space as LinearSpace>::E<'elem>);

    fn evaluate<'a>(
        &'a self,
        coeffs: &[FieldType<Self::Space>],
        result: &'a mut ElementType<'elem, Self::Space>,
    ) where
        'a: 'elem,
    {
        assert!(self.space().is_same(result.space()));
        assert_eq!(coeffs.len(), self.len());

        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.sum_into(coeff, elem);
        }
    }
}

pub struct DefaultFrame<'space: 'elem, 'elem, Space: LinearSpace> {
    data: Vec<<Space as LinearSpace>::E<'elem>>,
    space: &'space Space,
}

impl<'space: 'elem, 'elem, Space: LinearSpace> DefaultFrame<'space, 'elem, Space> {
    pub fn new(space: &'space Space) -> Self {
        Self {
            data: Vec::new(),
            space,
        }
    }
}

impl<'space: 'elem, 'elem, Space: LinearSpace> Frame<'space, 'elem>
    for DefaultFrame<'space, 'elem, Space>
{
    type Space = Space;

    type Iter<'iter> = std::slice::Iter<'iter, ElementType<'elem, Space>>
    where
        Self: 'elem,
        'elem: 'iter;

    type IterMut<'iter> = std::slice::IterMut<'iter, ElementType<'elem, Space>>
    where
        Self: 'elem,
        'elem: 'iter;

    fn get(&self, index: usize) -> Option<&ElementType<'elem, Self::Space>> {
        self.data.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<'elem, Self::Space>> {
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

    fn space(&self) -> &Self::Space {
        self.space
    }

    fn push(&mut self, elem: <Self::Space as LinearSpace>::E<'elem>) {
        self.data.push(elem)
    }
}
