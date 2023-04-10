//! Row vector layout.
//!
//! This layout describes a row vector whose elements
//! are in consecutive order in memory.

use crate::traits::*;
use crate::types::usize;

/// A type that describes a row vector with consecutive elements.
pub struct RowVector {
    dim: usize,
}

impl RowVector {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl LayoutType for RowVector {
    type IndexLayout = RowVector;

    #[inline]
    fn convert_1d_2d(&self, index: usize) -> (usize, usize) {
        (1, index)
    }

    #[inline]
    fn convert_2d_1d(&self, _row: usize, col: usize) -> usize {
        col
    }

    #[inline]
    fn convert_2d_raw(&self, _row: usize, col: usize) -> usize {
        col
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        index
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        (1, self.dim)
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        (self.dim, 1)
    }

    #[inline]
    fn number_of_elements(&self) -> usize {
        self.dim
    }

    #[inline]
    fn index_layout(&self) -> Self::IndexLayout {
        Self::IndexLayout::new(self.dim().1)
    }
}

impl BaseLayoutType for RowVector {
    fn from_dimension(dim: (usize, usize)) -> Self {
        assert_eq!(
            dim.0, 1,
            "Number of rows is {} but must be one for RowVector.",
            dim.0
        );
        Self { dim: dim.1 }
    }
}

impl VectorBaseLayoutType for RowVector {
    fn from_length(length: usize) -> Self {
        Self { dim: length }
    }
}

impl StridedLayoutType for RowVector {}
