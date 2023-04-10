//! Row vector with arbitrary stride.
//!
//! This layout describes a row vector with arbitrary
//! stride. The stride vector is by definition of the form
//! `(1, c)` for this type, where `c` is the distance of two
//! elements of the vector in memory.

use crate::traits::*;
use crate::types::usize;

use super::*;

/// A type that describes a row vector with arbitrary stride.
pub struct ArbitraryStrideRowVector {
    dim: usize,
    stride: usize,
}

impl ArbitraryStrideRowVector {
    pub fn new(dim: usize, stride: usize) -> Self {
        Self { dim, stride }
    }
}

impl LayoutType for ArbitraryStrideRowVector {
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
        col * self.stride
    }

    #[inline]
    fn convert_1d_raw(&self, index: usize) -> usize {
        index * self.stride
    }

    #[inline]
    fn dim(&self) -> (usize, usize) {
        (1, self.dim)
    }

    #[inline]
    fn stride(&self) -> (usize, usize) {
        (self.stride * self.dim, self.stride)
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

impl StridedLayoutType for ArbitraryStrideRowVector {}
