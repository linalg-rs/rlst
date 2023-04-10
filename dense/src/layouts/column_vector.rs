//! Column vector layout.
//!
//! This layout describes a column vector whose elements
//! are in consecutive order in memory.

use crate::traits::*;
use crate::types::usize;

/// A type that describes a column vector with consecutive elements.
pub struct ColumnVector {
    dim: usize,
}

impl ColumnVector {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl LayoutType for ColumnVector {
    type IndexLayout = ColumnVector;

    #[inline]
    fn convert_1d_2d(&self, index: usize) -> (usize, usize) {
        (index, 1)
    }

    #[inline]
    fn convert_2d_1d(&self, row: usize, _col: usize) -> usize {
        row
    }

    #[inline]
    fn convert_2d_raw(&self, row: usize, _col: usize) -> usize {
        row
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        index
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.dim, 1)
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        (1, self.dim)
    }

    #[inline]
    fn number_of_elements(&self) -> usize {
        self.dim
    }

    #[inline]
    fn index_layout(&self) -> Self::IndexLayout {
        Self::IndexLayout::new(self.dim().0)
    }
}

impl BaseLayoutType for ColumnVector {
    fn from_dimension(dim: (usize, usize)) -> Self {
        assert_eq!(
            dim.1, 1,
            "Number of columns is {} but must be one for ColumnVector.",
            dim.1
        );
        Self { dim: dim.0 }
    }
}

impl VectorBaseLayoutType for ColumnVector {
    fn from_length(length: usize) -> Self {
        Self { dim: length }
    }
}

impl StridedLayoutType for ColumnVector {}
