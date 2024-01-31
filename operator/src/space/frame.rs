//! A frame is a collection of elements of a space.

use crate::{Element, ElementType, FieldType, LinearSpace};

pub struct Frame<'space, Space: LinearSpace> {
    data: Vec<<Space as LinearSpace>::E<'space>>,
    space: &'space Space,
}

impl<'space, Space: LinearSpace> Frame<'space, Space> {
    pub fn new(space: &'space Space) -> Self {
        Self {
            data: Vec::new(),
            space,
        }
    }

    pub fn get(&self, index: usize) -> Option<&ElementType<'space, Space>> {
        self.data.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut ElementType<'space, Space>> {
        self.data.get_mut(index)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ElementType<'space, Space>> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, ElementType<'space, Space>> {
        self.data.iter_mut()
    }

    pub fn space(&self) -> &Space {
        self.space
    }

    pub fn push(&mut self, elem: <Space as LinearSpace>::E<'space>) {
        self.data.push(elem);
    }

    pub fn evaluate(&self, coeffs: &[FieldType<Space>], result: &mut ElementType<'space, Space>) {
        assert!(self.space.is_same(result.space()));
        assert_eq!(coeffs.len(), self.len());

        for (elem, coeff) in self.data.iter().zip(coeffs.iter().copied()) {
            result.sum_into(coeff, elem);
        }
    }
}
