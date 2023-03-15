//! Column vector with arbitrary stride.
//!
//! This layout describes a column vector with arbitrary
//! stride. The stride vector is by definition of the form
//! `(r, 1)` for this type, where `r` is the distance of two
//! elements of the vector in memory.

use crate::traits::*;
use crate::types::IndexType;

use super::*;

/// A type that describes a column vector with arbitrary stride.
pub struct ArbitraryStrideColumnVector {
    dim: IndexType,
    stride: IndexType,
}

impl ArbitraryStrideColumnVector {
    pub fn new(dim: IndexType, stride: IndexType) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for ArbitraryStrideColumnVector {
    type IndexLayout = ColumnVector;

    #[inline]
    fn convert_1d_2d(&self, index: IndexType) -> (IndexType, IndexType) {
        (index, 1)
    }

    #[inline]
    fn convert_2d_1d(&self, row: IndexType, _col: IndexType) -> IndexType {
        row
    }

    #[inline]
    fn convert_2d_raw(&self, row: IndexType, _col: IndexType) -> IndexType {
        row * self.stride
    }

    #[inline]
    fn convert_1d_raw(&self, index: IndexType) -> IndexType {
        index * self.stride
    }

    #[inline]
    fn dim(&self) -> (IndexType, IndexType) {
        (self.dim, 1)
    }

    #[inline]
    fn stride(&self) -> (IndexType, IndexType) {
        (self.stride, 1)
    }

    #[inline]
    fn number_of_elements(&self) -> IndexType {
        self.dim
    }

    #[inline]
    fn index_layout(&self) -> Self::IndexLayout {
        Self::IndexLayout::new(self.dim().0)
    }
}

impl StridedLayoutType for ArbitraryStrideColumnVector {}
