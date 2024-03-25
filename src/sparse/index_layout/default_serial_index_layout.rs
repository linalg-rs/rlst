//! Defailt serial index layout

use crate::dense::types::{RlstError, RlstResult};
use crate::sparse::traits::index_layout::IndexLayout;

/// Default serial index layout
pub struct DefaultSerialIndexLayout {
    size: usize,
}

impl DefaultSerialIndexLayout {
    /// Create new
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl IndexLayout for DefaultSerialIndexLayout {
    fn number_of_local_indices(&self) -> usize {
        self.number_of_global_indices()
    }

    fn local_range(&self) -> (usize, usize) {
        (0, self.size)
    }

    fn number_of_global_indices(&self) -> usize {
        self.size
    }

    fn index_range(&self, rank: usize) -> RlstResult<(usize, usize)> {
        if rank == 0 {
            Ok((0, self.size))
        } else {
            Err(RlstError::MpiRankError(rank as i32))
        }
    }

    fn local2global(&self, index: usize) -> Option<usize> {
        if index < self.number_of_local_indices() {
            Some(index)
        } else {
            None
        }
    }

    fn global2local(&self, rank: usize, index: usize) -> Option<usize> {
        if rank == 0 && index < self.number_of_global_indices() {
            Some(index)
        } else {
            None
        }
    }

    fn rank_from_index(&self, index: usize) -> Option<usize> {
        if index < self.number_of_global_indices() {
            Some(0)
        } else {
            None
        }
    }
}
