use crate::traits::index_layout::IndexLayout;
use mpi::traits::Communicator;
use rlst_common::types::{IndexType, RlstResult};

pub struct DefaultMpiIndexLayout<'a, C: Communicator> {
    size: IndexType,
    my_rank: IndexType,
    counts: Vec<IndexType>,
    comm: &'a C,
}

impl<'a, C: Communicator> DefaultMpiIndexLayout<'a, C> {
    pub fn new(size: IndexType, comm: &'a C) -> Self {
        let comm_size = comm.size() as IndexType;

        assert!(
            comm_size > 0,
            "Group size is zero. At least one process needs to be in the group."
        );
        let my_rank = comm.rank() as IndexType;
        let mut counts = vec![0 as IndexType; 1 + comm_size as usize];

        // The following code computes what index is on what rank. No MPI operation necessary.
        // Each process computes it from its own rank and the number of MPI processes in
        // the communicator

        if size <= comm_size {
            // If we have fewer indices than ranks simply
            // give one index to each rank until filled up.
            // Then fill the rest with None.

            for index in 0..size {
                counts[index] = index;
            }

            for index in size..comm_size {
                counts[index] = size;
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

            let chunk = size / comm_size;
            let remainder = size % comm_size;
            let mut count = 0;
            let mut new_count;

            for index in 0..comm_size {
                if index < remainder {
                    // Add one remainder index to the first
                    // indices.
                    new_count = count + chunk + 1;
                } else {
                    // When the remainder is used up just
                    // add chunk size indices to each rank.
                    new_count = count + chunk;
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
    pub fn comm(&self) -> &C {
        self.comm
    }
}

impl<'a, C: Communicator> IndexLayout for DefaultMpiIndexLayout<'a, C> {
    fn index_range(&self, rank: IndexType) -> RlstResult<(IndexType, IndexType)> {
        if rank < self.comm.size() as IndexType {
            Ok((self.counts[rank], self.counts[1 + rank]))
        } else {
            Err(rlst_common::types::RlstError::MpiRankError(rank as i32))
        }
    }

    fn local_range(&self) -> (IndexType, IndexType) {
        self.index_range(self.my_rank).unwrap()
    }

    fn number_of_local_indices(&self) -> IndexType {
        self.counts[1 + self.my_rank] - self.counts[self.my_rank]
    }

    fn number_of_global_indices(&self) -> IndexType {
        self.size
    }

    fn local2global(&self, index: IndexType) -> Option<IndexType> {
        if index < self.number_of_local_indices() {
            Some(self.counts[self.my_rank] + index)
        } else {
            None
        }
    }

    fn global2local(&self, rank: IndexType, index: IndexType) -> Option<IndexType> {
        if let Ok(index_range) = self.index_range(rank) {
            if index >= index_range.1 {
                return None;
            }

            Some(index - index_range.0)
        } else {
            None
        }
    }

    fn rank_from_index(&self, index: IndexType) -> Option<IndexType> {
        for (count_index, &count) in self.counts[1..].iter().enumerate() {
            if index < count {
                return Some(count_index as IndexType);
            }
        }
        None
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use mpi;

    #[test]
    fn test_distributed_index_set() {
        let universe = mpi::initialize().unwrap();
        let world = universe.world();

        let index_layout = DefaultMpiIndexLayout::new(14, &world);

        // Test that the range is correct on rank 0
        assert_eq!(index_layout.index_range(0).unwrap(), (0, 14));

        // Test that the number of global indices is correct.
        assert_eq!(index_layout.number_of_global_indices(), 14);

        // Test that map works

        assert_eq!(index_layout.local2global(2).unwrap(), 2);

        // Return the correct process for an index

        assert_eq!(index_layout.rank_from_index(5).unwrap(), 0);
    }
}
