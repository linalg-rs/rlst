//! Column major layout.
//!
//! This layout describes a column major memory order
//! with column major 1d indexing. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;
use crate::types::usize;

/// A type that describes a matrix in column major format.
pub struct ColumnMajor {
    dim: (usize, usize),
}

impl ColumnMajor {
    pub fn new(dim: (usize, usize)) -> Self {
        Self { dim }
    }
}

impl LayoutType for ColumnMajor {
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

impl BaseLayoutType for ColumnMajor {
    fn from_dimension(dim: (usize, usize)) -> Self {
        Self { dim }
    }
}

impl MatrixBaseLayoutType for ColumnMajor {}
impl StridedLayoutType for ColumnMajor {}
