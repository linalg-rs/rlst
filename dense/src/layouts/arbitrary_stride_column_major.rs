//! Column major layout with arbitrary stride.
//!
//! This layout uses an arbitrary stride in memory with 1d indexing
//! in column major order. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;
use crate::types::usize;

use super::*;

/// A type that describes a matrix with arbitrary stride and column major indexing.
pub struct ArbitraryStrideColumnMajor {
    dim: (usize, usize),
    stride: (usize, usize),
}

impl ArbitraryStrideColumnMajor {
    pub fn new(dim: (usize, usize), stride: (usize, usize)) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for ArbitraryStrideColumnMajor {
    type IndexLayout = ColumnMajor;

    #[inline]
    fn convert_1d_2d(&self, index: usize) -> (usize, usize) {
        (index % self.dim.0, index / self.dim.0)
    }

    #[inline]
    fn convert_2d_1d(&self, row: usize, col: usize) -> usize {
        col * self.dim.0 + row
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
        (1, self.dim.0)
    }

    #[inline]
    fn number_of_elements(&self) -> usize {
        self.dim.0 * self.dim.1
    }

    #[inline]
    fn index_layout(&self) -> Self::IndexLayout {
        Self::IndexLayout::new(self.dim())
    }
}

impl StridedLayoutType for ArbitraryStrideColumnMajor {}
