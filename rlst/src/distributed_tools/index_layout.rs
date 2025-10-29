//! Definition of Index Layouts.
//!
//! An [IndexLayout] specified how degrees of freedom are distributed among processes.
//! We always assume that a process has a contiguous set of degrees of freedom.

use crate::distributed_tools::all_to_allv;

use itertools::Itertools;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};

// An index layout specifying index ranges on each rank.
//
/// This index layout assumes a contiguous set of indices
/// starting with the first n0 indices on rank 0, the next n1 indices on rank 1, etc.
pub struct IndexLayout<'a, C: Communicator> {
    scan: Vec<usize>,
    comm: &'a C,
}

impl<C: Communicator> std::fmt::Debug for IndexLayout<'_, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IndexLayout with {} global indices and {} local indices.",
            self.number_of_global_indices(),
            self.number_of_local_indices()
        )
    }
}

impl<C: Communicator> Clone for IndexLayout<'_, C> {
    fn clone(&self) -> Self {
        Self {
            scan: self.scan.clone(),
            comm: self.comm,
        }
    }
}

impl<'a, C: Communicator> IndexLayout<'a, C> {
    /// Create a new index layout.
    ///
    /// `scan` is a vector of cumulative counts of indices of all previous ranks.
    pub fn new(scan: Vec<usize>, comm: &'a C) -> Self {
        Self { scan, comm }
    }

    /// Create an index layout with equidistributed chunks.
    ///
    /// A single chunk is a contiguous number of indices that should remain on the same process,
    /// e.g. `chunk_size=3` would be used to create an index layout for a number of points with 3 indices each.
    /// `nchunks` is the total number of chunks across all processes.
    /// The total number of indices is therefore `nchunks * chunk_size`.
    /// Chunks are distributed as equally as possible across the processes with the remainder distributed to the first few processes.
    pub fn from_equidistributed_chunks(nchunks: usize, chunk_size: usize, comm: &'a C) -> Self {
        let nindices = nchunks * chunk_size;
        let comm_size = comm.size() as usize;

        assert!(
            comm_size > 0,
            "Group size is zero. At least one process needs to be in the group."
        );
        let mut scan = vec![0; 1 + comm_size];

        // The following code computes what index is on what rank. No MPI operation necessary.
        // Each process computes it from its own rank and the number of MPI processes in
        // the communicator

        if nchunks <= comm_size {
            // If we have fewer chunks than ranks simply
            // give chunk_size indices to each rank until filled up.
            // Then fill the rest with None.

            for (index, item) in scan.iter_mut().enumerate().take(nchunks) {
                *item = index * chunk_size;
            }

            for item in scan.iter_mut().take(comm_size).skip(nchunks) {
                *item = nindices;
            }

            scan[comm_size] = nindices;
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
                scan[1 + index] = new_count;
                count = new_count;
            }
        }
        Self { scan, comm }
    }

    /// Create an index layout from each process reporting its own number of indices.
    pub fn from_local_counts(number_of_local_indices: usize, comm: &'a C) -> Self {
        let size = comm.size() as usize;
        let mut scan = vec![0; size + 1];
        comm.all_gather_into(&number_of_local_indices, &mut scan[1..]);
        for i in 1..=size {
            scan[i] += scan[i - 1];
        }
        Self { scan, comm }
    }

    /// The cumulative sum of indices over the ranks.
    ///
    /// The number of indices on rank i scan[1 + i] - scan[i].
    /// The last entry is the total number of indices.
    pub fn scan(&self) -> &[usize] {
        &self.scan
    }

    /// The local index range. If there is no local index
    /// the left and right bound are identical.
    pub fn local_range(&self) -> (usize, usize) {
        let scan = self.scan();
        (
            scan[self.comm().rank() as usize],
            scan[1 + self.comm().rank() as usize],
        )
    }

    /// The number of global indices.
    pub fn number_of_global_indices(&self) -> usize {
        *self.scan().last().unwrap()
    }

    /// The number of local indicies, that is the amount of indicies
    /// on my process.
    pub fn number_of_local_indices(&self) -> usize {
        let scan = self.scan();
        scan[1 + self.comm().rank() as usize] - scan[self.comm().rank() as usize]
    }

    /// Index range on a given process.
    pub fn index_range(&self, rank: usize) -> Option<(usize, usize)> {
        let scan = self.scan();
        if rank < self.comm().size() as usize {
            Some((scan[rank], scan[1 + rank]))
        } else {
            None
        }
    }

    /// Convert continuous (0, n) indices to actual indices.
    ///
    /// Assume that the local range is (30, 40). Then this method
    /// will map (0,10) -> (30, 40).
    /// It returns ```None``` if ```index``` is out of bounds.
    pub fn local2global(&self, index: usize) -> Option<usize> {
        let rank = self.comm().rank() as usize;
        if index < self.number_of_local_indices() {
            Some(self.scan()[rank] + index)
        } else {
            None
        }
    }

    /// Convert global index to local index on a given rank.
    /// Returns ```None``` if index does not exist on rank.
    pub fn global2local(&self, rank: usize, index: usize) -> Option<usize> {
        if let Some(index_range) = self.index_range(rank) {
            if index >= index_range.1 || index < index_range.0 {
                return None;
            }

            Some(index - index_range.0)
        } else {
            None
        }
    }

    /// Get the rank of a given index.
    pub fn rank_from_index(&self, index: usize) -> Option<usize> {
        for (count_index, &count) in self.scan()[1..].iter().enumerate() {
            if index < count {
                return Some(count_index);
            }
        }
        None
    }

    /// Get an array with the number of indices on each rank.
    pub fn local_counts(&self) -> Vec<usize> {
        self.scan()
            .iter()
            .tuple_windows()
            .map(|(start, end)| end - start)
            .collect()
    }

    /// Remap indices from one layout to another.
    pub fn remap<T: Equivalence>(&self, other: &IndexLayout<'a, C>, data: &[T]) -> Vec<T> {
        assert_eq!(data.len(), self.number_of_local_indices());
        assert_eq!(
            self.number_of_global_indices(),
            other.number_of_global_indices()
        );

        let my_range = self.local_range();

        let other_bins = (0..other.comm().size() as usize)
            .map(|rank| other.index_range(rank).unwrap().0)
            .collect_vec();

        let sorted_keys = (my_range.0..my_range.1).collect_vec();

        let counts = super::array_tools::sort_to_bins(&sorted_keys, &other_bins);

        all_to_allv(other.comm(), &counts, data).1
    }

    /// Return the communicator.
    pub fn comm(&self) -> &C {
        self.comm
    }

    /// Check if two index layouts are pointer equal.
    pub fn is_same(&self, other: &IndexLayout<'a, C>) -> bool {
        std::ptr::eq(self, other)
    }
}
