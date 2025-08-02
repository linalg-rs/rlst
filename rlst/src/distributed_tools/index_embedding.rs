//! An index embedding handles a distribution of subindices from a global index set.
//! Consider that we have a global index set with a number of indices on each process.
//! We want to select a subset of these indices on each process. The `IndexEmbedding` struct
//! efficiently handles mapping between indices relative to this subset and the global index set.

use std::{collections::HashMap, rc::Rc};

use itertools::izip;
use mpi::traits::Communicator;

use super::IndexLayout;

/// Create a new embedded indexing
pub struct IndexEmbedding<'a, C: Communicator> {
    global_layout: Rc<IndexLayout<'a, C>>,
    embedded_index_subset: Vec<usize>,
    embedded_layout: Rc<IndexLayout<'a, C>>,
    local_to_embedded_index: HashMap<usize, usize>,
}

impl<'a, C: Communicator> IndexEmbedding<'a, C> {
    /// Create a new index embedding.
    ///
    /// Note. Each index in `embedded_index_subset` must be unique.
    pub fn new(
        global_layout: Rc<IndexLayout<'a, C>>,
        embedded_index_subset: &[usize],
        comm: &'a C,
    ) -> Self {
        // Let us setup an index layout for the subset.

        let embedded_layout = Rc::new(IndexLayout::from_local_counts(
            embedded_index_subset.len(),
            comm,
        ));

        // We now need to setup the maps between local indexing and global indexing.
        // From embedded to local is easy. This is just the index itself. From local to embedded is a bit more tricky.
        // We need a HashMap for this.

        let local_to_embedded_index: HashMap<usize, usize> = embedded_index_subset
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, i))
            .collect();

        // That's it. Return the struct.

        Self {
            global_layout,
            embedded_index_subset: embedded_index_subset.to_vec(),
            embedded_layout,
            local_to_embedded_index,
        }
    }

    /// Return the embedded index layout
    pub fn embedded_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.embedded_layout.clone()
    }

    /// Return the global layout
    pub fn global_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.global_layout.clone()
    }

    /// Map an index with respect to the embedded indexing to the corresponding local index
    pub fn embedded_index_to_local_index(&self, embedded_index: usize) -> usize {
        self.embedded_index_subset[embedded_index]
    }

    /// Map an index with respect to local indexing to the corresponding embedded index
    ///
    /// Returns None if no corresponding embedded index exists.
    pub fn local_index_to_embedded_index(&self, local_index: usize) -> Option<usize> {
        self.local_to_embedded_index.get(&local_index).copied()
    }

    /// Map an embedded index to the corresponding global index
    pub fn embedded_index_to_global_index(&self, embedded_index: usize) -> usize {
        self.global_layout
            .local2global(self.embedded_index_to_local_index(embedded_index))
            .unwrap()
    }

    /// Map a global index to the corresponding embedded index
    pub fn global_index_to_embedded_index(&self, global_index: usize) -> Option<usize> {
        let rank = self.global_layout.comm().rank() as usize;
        self.local_index_to_embedded_index(self.global_layout.global2local(rank, global_index)?)
    }

    /// Embed a data vector from an embedded indexing to a local indexing.
    ///
    /// Let there be `m` embedded indices and `n` local indices. A vector of length `m * chunk_size` is
    /// copied into a vector of length `n * chunk_size` where the data is copied from the embedded indices
    /// to the corresponding local index positions. The values not contained in the embedded indices are set to
    /// the default value of the data type.
    pub fn embed_data<T: Default + Copy>(
        &self,
        data: &[T],
        out_vector: &mut [T],
        chunk_size: usize,
    ) {
        for (local_index, chunk) in
            izip!(self.embedded_index_subset.iter(), data.chunks(chunk_size))
        {
            let local_index = self.local_index_to_embedded_index(*local_index).unwrap();
            let local_start_index = local_index * chunk_size;
            let local_end_index = local_start_index + chunk_size;
            out_vector[local_start_index..local_end_index].copy_from_slice(chunk);
        }
    }

    /// Extract embedded data from a local vector.
    ///
    /// Given a vector of length `n * chunk_size` where `n` is the number of local indices, extract the data
    /// associated with the `m` embedded indices. The data is copied into a vector of length `m * chunk_size`
    /// with ordering given by the embedded indices.
    pub fn extract_embedded_data<T: Default + Copy>(
        &self,
        data: &[T],
        chunk_size: usize,
    ) -> Vec<T> {
        let mut extracted_data =
            vec![T::default(); self.embedded_layout.number_of_local_indices() * chunk_size];

        for (local_index, chunk) in izip!(
            self.embedded_index_subset.iter(),
            extracted_data.chunks_mut(chunk_size)
        ) {
            chunk.copy_from_slice(&data[local_index * chunk_size..(1 + local_index) * chunk_size]);
        }

        extracted_data
    }
}
