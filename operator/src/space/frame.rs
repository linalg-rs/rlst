//! A frame is a collection of elements of a space.

use crate::{Element, ElementType, FieldType, LinearSpace};

pub trait Frame<'space> {
    type Space: LinearSpace;

    type Iter<'a>: std::iter::Iterator<Item = &'a ElementType<'space, Self::Space>>
    where
        Self::Space: 'space,
        Self: 'a,
        'space: 'a;

    type IterMut<'a>: std::iter::Iterator<Item = &'a mut ElementType<'space, Self::Space>>
    where
        Self::Space: 'space,
        Self: 'a,
        'space: 'a;

    fn get(&self, index: usize) -> Option<&ElementType<'space, Self::Space>>;

    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<'space, Self::Space>>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn iter(&self) -> Self::Iter<'_>;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn space(&self) -> &Self::Space;

    fn push(&mut self, elem: <Self::Space as LinearSpace>::E<'space>);

    fn evaluate<'a>(
        &'a self,
        coeffs: &'a [FieldType<Self::Space>],
        result: &'a mut ElementType<'space, Self::Space>,
    ) {
        assert!(self.space().is_same(result.space()));
        assert_eq!(coeffs.len(), self.len());

        for (elem, coeff) in self.iter().zip(coeffs.iter().copied()) {
            result.sum_into(coeff, elem);
        }
    }
}

pub struct DefaultFrame<'space, Space: LinearSpace> {
    data: Vec<<Space as LinearSpace>::E<'space>>,
    space: &'space Space,
}

impl<'space, Space: LinearSpace> DefaultFrame<'space, Space> {
    pub fn new(space: &'space Space) -> Self {
        Self {
            data: Vec::new(),
            space,
        }
    }
}

impl<'space, Space: LinearSpace> Frame<'space> for DefaultFrame<'space, Space> {
    type Space = Space;

    type Iter<'a> = std::slice::Iter<'a, ElementType<'space, Space>>
    where
        Self::Space: 'a,
        Self: 'a,
        'space: 'a;

    type IterMut<'a> = std::slice::IterMut<'a, ElementType<'space, Space>>
    where
        Self::Space: 'a,
        Self: 'a,
        'space: 'a;

    fn get(&self, index: usize) -> Option<&ElementType<'space, Self::Space>> {
        self.data.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<'space, Self::Space>> {
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

    fn push(&mut self, elem: <Self::Space as LinearSpace>::E<'space>) {
        self.data.push(elem)
    }
}
