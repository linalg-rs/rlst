//! Column vector with arbitrary stride.
//!
//! This layout describes a column vector with arbitrary
//! stride. The stride vector is by definition of the form
//! `(r, 1)` for this type, where `r` is the distance of two
//! elements of the vector in memory.

use crate::traits::*;
use crate::types::usize;

use super::*;

/// A type that describes a column vector with arbitrary stride.
pub struct ArbitraryStrideColumnVector {
    dim: usize,
    stride: usize,
}

impl ArbitraryStrideColumnVector {
    pub fn new(dim: usize, stride: usize) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for ArbitraryStrideColumnVector {
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
        row * self.stride
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        index * self.stride
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.dim, 1)
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        (self.stride, self.stride * self.dim)
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

impl StridedLayoutType for ArbitraryStrideColumnVector {}
