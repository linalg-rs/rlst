//! Row major layout with arbitrary stride.
//!
//! This layout uses an arbitrary stride in memory with 1d indexing
//! in row major order. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;

pub struct DefaultLayout {
    dim: (usize, usize),
    stride: (usize, usize),
}

impl DefaultLayout {
    pub fn new(dim: (usize, usize), stride: (usize, usize)) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for DefaultLayout {
    #[inline]
    fn convert_1d_2d(&self, index: usize) -> (usize, usize) {
        (index / self.dim.1, index % self.dim.1)
    }

    #[inline]
    fn convert_2d_1d(&self, row: usize, col: usize) -> usize {
        row * self.dim.1 + col
    }

    #[inline]
    fn convert_2d_raw(&self, row: usize, col: usize) -> usize {
        self.stride.0 * row + self.stride.1 * col
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        let (row, col) = self.convert_1d_2d(index);
        self.convert_2d_raw(row, col)
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        self.dim
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        self.stride
    }

    #[inline]
    fn number_of_elements(&self) -> usize {
        self.dim.0 * self.dim.1
    }

    fn from_dimension(dim: (usize, usize), stride: (usize, usize)) -> Self {
        Self { dim, stride }
    }
}
