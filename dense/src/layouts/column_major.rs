//! Column major layout.
//!
//! This layout describes a column major memory order
//! with column major 1d indexing. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;
use crate::types::IndexType;

/// A type that describes a matrix in column major format.
pub struct ColumnMajor {
    dim: (IndexType, IndexType),
}

impl ColumnMajor {
    pub fn new(dim: (IndexType, IndexType)) -> Self {
        Self { dim }
    }
}

impl LayoutType for ColumnMajor {
    type IndexLayout = ColumnMajor;

    #[inline]
    fn convert_1d_2d(&self, index: IndexType) -> (IndexType, IndexType) {
        (index % self.dim.0, index / self.dim.0)
    }

    #[inline]
    fn convert_2d_1d(&self, row: IndexType, col: IndexType) -> IndexType {
        col * self.dim.0 + row
    }

    #[inline]
    fn convert_2d_raw(&self, row: IndexType, col: IndexType) -> IndexType {
        self.convert_2d_1d(row, col)
    }

    #[inline]
    fn convert_1d_raw(&self, index: IndexType) -> IndexType {
        index
    }

    #[inline]
    fn dim(&self) -> (IndexType, IndexType) {
        self.dim
    }

    #[inline]
    fn stride(&self) -> (IndexType, IndexType) {
        (1, self.dim.0)
    }

    #[inline]
    fn number_of_elements(&self) -> IndexType {
        self.dim.0 * self.dim.1
    }

    #[inline]
    fn index_layout(&self) -> Self::IndexLayout {
        Self::IndexLayout::new(self.dim())
    }
}

impl BaseLayoutType for ColumnMajor {
    fn from_dimension(dim: (IndexType, IndexType)) -> Self {
        Self { dim }
    }
}

impl MatrixBaseLayoutType for ColumnMajor {}
impl StridedLayoutType for ColumnMajor {}
