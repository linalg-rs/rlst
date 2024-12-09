//! Definition of CSR matrices.

use std::collections::HashMap;

use crate::sparse::index_layout::DefaultDistributedIndexLayout;
use crate::sparse::sparse_mat::csr_mat::CsrMatrix;
use crate::sparse::sparse_mat::SparseMatType;
use crate::sparse::traits::index_layout::IndexLayout;
use bempp_ghost::GhostCommunicator;
use itertools::{izip, Itertools};

use crate::sparse::distributed_vector::DistributedVector;
use mpi::traits::{Communicator, Equivalence, Root};

use crate::dense::traits::Shape;
use crate::dense::traits::{RawAccess, RawAccessMut};
use crate::dense::types::RlstScalar;

/// Distributed CSR matrix
pub struct DistributedCsrMatrix<'a, T: RlstScalar + Equivalence, C: Communicator> {
    mat_type: SparseMatType,
    shape: [usize; 2],
    local_matrix: CsrMatrix<T>,
    global_indices: Vec<usize>,
    local_dof_count: usize,
    domain_layout: &'a DefaultDistributedIndexLayout<'a, C>,
    range_layout: &'a DefaultDistributedIndexLayout<'a, C>,
    domain_ghosts: GhostCommunicator<usize>,
}

impl<'a, T: RlstScalar + Equivalence, C: Communicator> DistributedCsrMatrix<'a, T, C> {
    /// Create new
    pub fn new(
        shape: [usize; 2],
        indices: Vec<usize>,
        indptr: Vec<usize>,
        data: Vec<T>,
        domain_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        range_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        comm: &'a C,
    ) -> Self {
        let my_rank = comm.rank() as usize;

        let domain_ghost_dofs: Vec<usize> = indices
            .iter()
            .unique()
            .copied()
            .filter(|&dof| domain_layout.rank_from_index(dof).unwrap() != my_rank)
            .collect();

        let ranks = domain_ghost_dofs
            .iter()
            .map(|dof| domain_layout.rank_from_index(*dof).unwrap())
            .collect::<Vec<_>>();

        let domain_ghosts = GhostCommunicator::new(&domain_ghost_dofs, &ranks, comm);
        let local_dof_count =
            domain_layout.number_of_local_indices() + domain_ghosts.total_receive_count;
        let mut index_mapper = HashMap::new();

        // We need to transform the indices vector of the CSR matrix from global indexing to
        // local indexing. For this we assume that the input vector has the format
        // [local_indices..., ghost_indices]. So we map global indices to fit this format.
        // To do this we create a hash map that takes the global indices and maps to the new
        // local indexing.

        let mut count: usize = 0;
        for index in domain_layout.local_range().0..domain_layout.local_range().1 {
            index_mapper.insert(index, count);
            count += 1;
        }
        for index in &domain_ghosts.receive_indices {
            index_mapper.insert(*index, count);
            count += 1;
        }

        // The hash map is created. We now iterate through the indices vector of the sparse matrix
        // to change the indexing.

        let mapped_indices = indices
            .iter()
            .map(|elem| *index_mapper.get(elem).unwrap())
            .collect::<Vec<_>>();

        Self {
            mat_type: SparseMatType::Csr,
            shape,
            local_matrix: CsrMatrix::new(
                [indptr.len() - 1, shape[1]],
                mapped_indices,
                indptr,
                data,
            ),
            global_indices: indices,
            local_dof_count,
            domain_layout,
            range_layout,
            domain_ghosts,
        }
    }

    /// Matrix type
    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    /// Local shape
    pub fn local_shape(&self) -> [usize; 2] {
        self.local_matrix.shape()
    }

    /// Column indices
    pub fn indices(&self) -> &[usize] {
        &self.global_indices
    }

    /// Indices at which each row starts
    pub fn indptr(&self) -> &[usize] {
        self.local_matrix.indptr()
    }

    /// Matrix entries
    pub fn data(&self) -> &[T] {
        self.local_matrix.data()
    }

    /// Domain layout
    pub fn domain_layout(&self) -> &'a DefaultDistributedIndexLayout<'a, C> {
        self.domain_layout
    }

    /// Range layout
    pub fn range_layout(&self) -> &'a DefaultDistributedIndexLayout<'a, C> {
        self.range_layout
    }

    /// Create from root
    pub fn from_serial(
        root: usize,
        domain_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        range_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        comm: &'a C,
    ) -> Self {
        let root_process = comm.process_at_rank(root as i32);

        let my_index_range = range_layout.local_range();
        let my_number_of_rows = my_index_range.1 - my_index_range.0;

        let mut csr_data;
        let mut csr_indices;
        let mut csr_indptr;
        let mut shape;
        let mut my_data_count: i32 = 0;

        shape = vec![0; 2];

        // Receive the number of data entries.
        root_process.scatter_into(&mut my_data_count);

        // Now make space for the matrix data.

        csr_data = vec![T::zero(); my_data_count as usize];
        csr_indices = vec![0; my_data_count as usize];
        csr_indptr = vec![0; 1 + my_number_of_rows];

        // Get the matrix data
        root_process.scatter_varcount_into(csr_data.as_mut_slice());
        root_process.scatter_varcount_into(csr_indices.as_mut_slice());
        root_process.scatter_varcount_into(&mut csr_indptr.as_mut_slice()[..my_number_of_rows]);
        root_process.broadcast_into(shape.as_mut_slice());

        // We need to fix the index pointer as locally it needs to start at 0. But
        // the communicated data is offset by where the index pointer was on the
        // root process.
        let first_elem = csr_indptr[0];
        csr_indptr[..my_number_of_rows]
            .iter_mut()
            .for_each(|elem| *elem -= first_elem);

        csr_indptr[my_number_of_rows] = my_data_count as usize;

        // Once everything is received we can finally create the matrix object.

        Self::new(
            [shape[0], shape[1]],
            csr_indices,
            csr_indptr,
            csr_data,
            domain_layout,
            range_layout,
            comm,
        )
    }

    /// Create from root
    pub fn from_serial_root(
        csr_mat: CsrMatrix<T>,
        domain_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        range_layout: &'a DefaultDistributedIndexLayout<'a, C>,
        comm: &'a C,
    ) -> Self {
        let size = comm.size() as usize;
        let root_process = comm.this_process();

        let my_index_range = range_layout.local_range();
        let my_number_of_rows = my_index_range.1 - my_index_range.0;

        let mut csr_data;
        let mut csr_indices;
        let mut csr_indptr;
        let mut shape;

        // Need to compute how much data to send to each process.

        let mut my_data_count: i32 = 0;

        let mut counts: Vec<i32> = vec![0; size];

        for (rank, item) in counts.iter_mut().enumerate() {
            let local_index_range = range_layout.index_range(rank).unwrap();
            *item = (csr_mat.indptr()[local_index_range.1] - csr_mat.indptr()[local_index_range.0])
                as i32;
        }

        // Send around how much data is received by everyone.

        root_process.scatter_into_root(&counts, &mut my_data_count);

        // Every process now knows how much data it gets. Now compute the displacements.

        let mut count = 0;
        let mut displacements = Vec::<i32>::with_capacity(size);

        for n in &counts {
            displacements.push(count);
            count += n;
        }

        csr_data = vec![T::zero(); my_data_count as usize];
        csr_indices = vec![0; my_data_count as usize];
        csr_indptr = vec![0; 1 + my_number_of_rows];
        shape = vec![csr_mat.shape()[0], csr_mat.shape()[1]];

        // The following code computes the counts and displacements for the indexptr vector.

        let mut idxptrcount: Vec<i32> = vec![0; size];
        let mut idxptrdisplacements: Vec<i32> = vec![0; size];

        for rank in 0..size {
            let local_index_range = range_layout.index_range(rank).unwrap();
            idxptrcount[rank] = (local_index_range.1 - local_index_range.0) as i32;
            idxptrdisplacements[rank] = local_index_range.0 as i32;
        }

        // We now scatter the csr matrix data to the processes.

        let data_partition = mpi::datatype::Partition::new(
            csr_mat.data(),
            counts.as_slice(),
            displacements.as_slice(),
        );

        let indices_partition = mpi::datatype::Partition::new(
            csr_mat.indices(),
            counts.as_slice(),
            displacements.as_slice(),
        );

        let idx_partition = mpi::datatype::Partition::new(
            csr_mat.indptr(),
            idxptrcount.as_slice(),
            idxptrdisplacements.as_slice(),
        );

        // Send everything around
        root_process.scatter_varcount_into_root(&data_partition, csr_data.as_mut_slice());
        root_process.scatter_varcount_into_root(&indices_partition, csr_indices.as_mut_slice());
        root_process.scatter_varcount_into_root(
            &idx_partition,
            &mut csr_indptr.as_mut_slice()[..my_number_of_rows],
        );
        root_process.broadcast_into(shape.as_mut_slice());

        // We need to fix the index pointer as locally it needs to start at 0. But
        // the communicated data is offset by where the index pointer was on the
        // root process.
        let first_elem = csr_indptr[0];
        csr_indptr[..my_number_of_rows]
            .iter_mut()
            .for_each(|elem| *elem -= first_elem);

        csr_indptr[my_number_of_rows] = my_data_count as usize;

        // Once everything is received we can finally create the matrix object.

        Self::new(
            [shape[0], shape[1]],
            csr_indices,
            csr_indptr,
            csr_data,
            domain_layout,
            range_layout,
            comm,
        )
    }

    /// Matrix multiplication
    pub fn matmul<'b>(
        &self,
        alpha: T,
        x: &DistributedVector<'b, T, C>,
        beta: T,
        y: &mut DistributedVector<'b, T, C>,
    ) {
        // Create a vector that combines local dofs and ghosts

        let my_rank = self.domain_layout.comm().rank() as usize;

        let out_values = {
            let mut out_values = Vec::<T>::with_capacity(self.domain_ghosts.total_send_count());
            let out_buff: &mut [T] =
                unsafe { std::mem::transmute(out_values.spare_capacity_mut()) };

            for (out, out_index) in
                izip!(out_buff.iter_mut(), self.domain_ghosts.send_indices.iter())
            {
                *out = x.local().data()[self
                    .domain_layout()
                    .global2local(my_rank, *out_index)
                    .unwrap()];
            }

            unsafe { out_values.set_len(self.domain_ghosts.total_send_count()) };
            out_values
        };

        let local_vec = {
            let mut local_vec = Vec::<T>::with_capacity(self.local_dof_count);
            local_vec.extend(x.local().data().iter().copied());
            let ghost_data: &mut [T] =
                unsafe { std::mem::transmute(local_vec.spare_capacity_mut()) };

            // Prepare the values that are sent to other ranks

            self.domain_ghosts
                .forward_send_values(&out_values, ghost_data);
            unsafe { local_vec.set_len(self.local_dof_count) };
            local_vec
        };

        // Compute result
        self.local_matrix
            .matmul(alpha, local_vec.as_slice(), beta, y.local_mut().data_mut());
    }
}

impl<T: RlstScalar + Equivalence, C: Communicator> Shape<2> for DistributedCsrMatrix<'_, T, C> {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}
