//! Row major layout with arbitrary stride.
//!
//! This layout uses an arbitrary stride in memory with 1d indexing
//! in row major order. For further information on memory layouts
//! see [crate::traits::layout].

use crate::traits::*;
use crate::types::IndexType;

pub struct DefaultLayout {
    dim: (IndexType, IndexType),
    stride: (IndexType, IndexType),
}

impl DefaultLayout {
    pub fn new(dim: (IndexType, IndexType), stride: (IndexType, IndexType)) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for DefaultLayout {
    #[inline]
    fn convert_1d_2d(&self, index: IndexType) -> (IndexType, IndexType) {
        (index / self.dim.1, index % self.dim.1)
    }

    #[inline]
    fn convert_2d_1d(&self, row: IndexType, col: IndexType) -> IndexType {
        row * self.dim.1 + col
    }

    #[inline]
    fn convert_2d_raw(&self, row: IndexType, col: IndexType) -> IndexType {
        self.stride.0 * row + self.stride.1 * col
    }

    #[inline]
    fn convert_1d_raw(&self, index: IndexType) -> IndexType {
        let (row, col) = self.convert_1d_2d(index);
        self.convert_2d_raw(row, col)
    }

    #[inline]
    fn dim(&self) -> (IndexType, IndexType) {
        self.dim
    }

    #[inline]
    fn stride(&self) -> (IndexType, IndexType) {
        self.stride
    }

    #[inline]
    fn number_of_elements(&self) -> IndexType {
        self.dim.0 * self.dim.1
    }

    fn from_dimension(dim: (IndexType, IndexType), stride: (IndexType, IndexType)) -> Self {
        Self { dim, stride }
    }
}
