use crate::traits::index_layout::IndexLayout;
use rlst_dense::types::{RlstError, RlstResult};

pub struct DefaultSerialIndexLayout {
    size: usize,
}

impl DefaultSerialIndexLayout {
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
