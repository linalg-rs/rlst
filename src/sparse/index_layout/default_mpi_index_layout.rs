//! Default index layout
use crate::dense::types::RlstResult;
use crate::sparse::traits::index_layout::IndexLayout;
use mpi::traits::Communicator;

/// Default index layout
pub struct DefaultMpiIndexLayout<'a, C: Communicator> {
    size: usize,
    my_rank: usize,
    counts: Vec<usize>,
    comm: &'a C,
}

impl<'a, C: Communicator> DefaultMpiIndexLayout<'a, C> {
    /// Crate new
    pub fn new(nchunks: usize, chunk_size: usize, comm: &'a C) -> Self {
        let size = nchunks * chunk_size;
        let comm_size = comm.size() as usize;

        assert!(
            comm_size > 0,
            "Group size is zero. At least one process needs to be in the group."
        );
        let my_rank = comm.rank() as usize;
        let mut counts = vec![0; 1 + comm_size];

        // The following code computes what index is on what rank. No MPI operation necessary.
        // Each process computes it from its own rank and the number of MPI processes in
        // the communicator

        if nchunks <= comm_size {
            // If we have fewer chunks than ranks simply
            // give chunk_size indices to each rank until filled up.
            // Then fill the rest with None.

            for (index, item) in counts.iter_mut().enumerate().take(nchunks) {
                *item = index * chunk_size;
            }

            for item in counts.iter_mut().take(comm_size).skip(nchunks) {
                *item = size;
            }

            counts[comm_size] = size;
        } else {
            // We want to equally distribute the range
            // among the ranks. Assume that we have 12
            // indices and want to distribute among 5 ranks.
            // Then each rank gets 12 / 5 = 2 indices. However,
            // we have a remainder 12 % 5 = 2. Those two indices
            // are distributed among the first two ranks. So at
            // the end we have the distribution
            // 0 -> (0, 3)
            // 1 -> (3, 6)
            // 2 -> (6, 8)
            // 3 -> (8, 10)
            // 4 -> (10, 12)

            let chunks_per_rank = nchunks / comm_size;
            let remainder = nchunks % comm_size;
            let mut count = 0;
            let mut new_count;

            for index in 0..comm_size {
                if index < remainder {
                    // Add one remainder index to the first
                    // indices.
                    new_count = count + chunks_per_rank * chunk_size + chunk_size;
                } else {
                    // When the remainder is used up just
                    // add chunk size indices to each rank.
                    new_count = count + chunks_per_rank * chunk_size;
                }
                counts[1 + index] = new_count;
                count = new_count;
            }
        }

        Self {
            size,
            my_rank,
            counts,
            comm,
        }
    }
    /// MPI communicator
    pub fn comm(&self) -> &C {
        self.comm
    }
}

impl<'a, C: Communicator> IndexLayout for DefaultMpiIndexLayout<'a, C> {
    fn index_range(&self, rank: usize) -> RlstResult<(usize, usize)> {
        if rank < self.comm.size() as usize {
            Ok((self.counts[rank], self.counts[1 + rank]))
        } else {
            Err(crate::dense::types::RlstError::MpiRankError(rank as i32))
        }
    }

    fn local_range(&self) -> (usize, usize) {
        self.index_range(self.my_rank).unwrap()
    }

    fn number_of_local_indices(&self) -> usize {
        self.counts[1 + self.my_rank] - self.counts[self.my_rank]
    }

    fn number_of_global_indices(&self) -> usize {
        self.size
    }

    fn local2global(&self, index: usize) -> Option<usize> {
        if index < self.number_of_local_indices() {
            Some(self.counts[self.my_rank] + index)
        } else {
            None
        }
    }

    fn global2local(&self, rank: usize, index: usize) -> Option<usize> {
        if let Ok(index_range) = self.index_range(rank) {
            if index >= index_range.1 {
                return None;
            }

            Some(index - index_range.0)
        } else {
            None
        }
    }

    fn rank_from_index(&self, index: usize) -> Option<usize> {
        for (count_index, &count) in self.counts[1..].iter().enumerate() {
            if index < count {
                return Some(count_index);
            }
        }
        None
    }
}
