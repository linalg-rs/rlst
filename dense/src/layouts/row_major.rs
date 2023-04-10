//! Row major layout.
//!
//! This layout describes a row major memory order
//! with row major 1d indexing. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;
use crate::types::usize;

/// A type that describes a matrix in row major format.
pub struct RowMajor {
    dim: (usize, usize),
}

impl RowMajor {
    pub fn new(dim: (usize, usize)) -> Self {
        Self { dim }
    }
}

impl LayoutType for RowMajor {
    type IndexLayout = RowMajor;

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
        self.convert_2d_1d(row, col)
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        index
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        self.dim
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        (self.dim.1, 1)
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

impl BaseLayoutType for RowMajor {
    fn from_dimension(dim: (usize, usize)) -> Self {
        Self { dim }
    }
}

impl MatrixBaseLayoutType for RowMajor {}
impl StridedLayoutType for RowMajor {}
