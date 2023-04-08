//! Definition of CSR matrices.

use crate::ghost_communicator::GhostCommunicator;
use crate::index_layout::DefaultMpiIndexLayout;
use crate::sparse::csr_mat::CsrMatrix;
use crate::sparse::SparseMatType;
use crate::traits::index_layout::IndexLayout;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence, Root};

use rlst_common::types::{IndexType, Scalar};

pub struct MpiCsrMatrix<'a, T: Scalar + Equivalence, C: Communicator> {
    mat_type: SparseMatType,
    shape: (IndexType, IndexType),
    local_matrix: CsrMatrix<T>,
    domain_layout: &'a DefaultMpiIndexLayout<'a, C>,
    range_layout: &'a DefaultMpiIndexLayout<'a, C>,
    domain_ghosts: crate::ghost_communicator::GhostCommunicator,
    //range_ghost: crate::ghost_communicator::GhostCommunicator,
}

impl<'a, T: Scalar + Equivalence, C: Communicator> MpiCsrMatrix<'a, T, C> {
    pub fn new(
        shape: (IndexType, IndexType),
        indices: Vec<IndexType>,
        indptr: Vec<IndexType>,
        data: Vec<T>,
        domain_layout: &'a DefaultMpiIndexLayout<'a, C>,
        range_layout: &'a DefaultMpiIndexLayout<'a, C>,
        comm: &'a C,
    ) -> Self {
        let my_rank = comm.rank() as usize;

        let domain_ghost_dofs: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|&dof| domain_layout.rank_from_index(dof).unwrap() != my_rank)
            .collect();

        let domain_ghosts = GhostCommunicator::new(&domain_ghost_dofs, domain_layout, comm);

        Self {
            mat_type: SparseMatType::Csr,
            shape,
            local_matrix: CsrMatrix::new((indptr.len() - 1, shape.1), indices, indptr, data),
            domain_layout,
            range_layout,
            domain_ghosts,
        }
    }

    pub fn mat_type(&self) -> &SparseMatType {
        &self.mat_type
    }

    pub fn shape(&self) -> (IndexType, IndexType) {
        self.shape
    }

    pub fn local_shape(&self) -> (IndexType, IndexType) {
        self.local_matrix.shape()
    }

    pub fn indices(&self) -> &[IndexType] {
        &self.local_matrix.indices()
    }

    pub fn indptr(&self) -> &[IndexType] {
        &self.local_matrix.indptr()
    }

    pub fn data(&self) -> &[T] {
        &self.local_matrix.data()
    }

    pub fn from_csr(
        csr_mat: Option<CsrMatrix<T>>,
        domain_layout: &'a DefaultMpiIndexLayout<'a, C>,
        range_layout: &'a DefaultMpiIndexLayout<'a, C>,
        comm: &'a C,
    ) {
        let my_rank = comm.rank();
        let size = comm.size() as usize;
        let root_process = comm.process_at_rank(0);

        let my_index_range = range_layout.local_range();
        println!("{:#?}", my_index_range);
        let my_number_of_rows = my_index_range.1 - my_index_range.0;

        if my_rank == 1 {
            println!("Rank 1 index range. {:#?}", my_index_range);
        }

        if csr_mat.is_some() && my_rank != 0 {
            comm.abort(13); // Unknown error
        }

        if csr_mat.is_none() && my_rank == 0 {
            comm.abort(13);
        }

        // Need to compute how much data to send to each process.

        let mut my_data_count: usize = 0;

        if my_rank == 0 {
            let mut counts = vec![0 as i32; size];
            let csr_mat = csr_mat.unwrap();

            for rank in 0..size {
                let local_index_range = range_layout.index_range(rank).unwrap();
                counts[rank] = (csr_mat.indptr()[local_index_range.1]
                    - csr_mat.indptr()[local_index_range.0]) as i32;
            }

            println!("csr_mat.indptr {:#?}", csr_mat.indptr());

            // Scatter the data to the processes.

            root_process.scatter_into_root(&counts, &mut my_data_count);

            // Every process now knows how much data it gets. Now compute the displacements.

            let mut count = 0;
            let mut displacements = Vec::<i32>::with_capacity(size);

            for n in &counts {
                displacements.push(count);
                count += n;
            }

            let mut csr_data = vec![T::zero(); my_data_count];
            let mut csr_indices = vec![0 as usize; my_data_count];
            let mut csr_indptr = vec![0 as usize; 1 + my_number_of_rows];
            let shape = vec![0 as usize; 2];

            // We now scatter the csr matrix data to the processes.

            let partition = mpi::datatype::Partition::new(
                csr_mat.data(),
                counts.as_slice(),
                displacements.as_slice(),
            );

            root_process.scatter_varcount_into_root(&partition, csr_data.as_mut_slice());

            // The following scatters the indices to the processes.

            let partition = mpi::datatype::Partition::new(
                csr_mat.indices(),
                counts.as_slice(),
                displacements.as_slice(),
            );

            root_process.scatter_varcount_into_root(&partition, csr_indices.as_mut_slice());

            // We now need to send around the index pointers. For that we need the index pointer counts
            // and index pointer displacements.

            let mut idxptrcount = vec![0 as i32; size];
            let mut idxptrdisplacements = vec![0 as i32; size];

            for rank in 0..size {
                let local_index_range = range_layout.index_range(rank).unwrap();
                idxptrcount[rank] = 1 + (local_index_range.1 - local_index_range.0) as i32;
                idxptrdisplacements[rank] = local_index_range.0 as i32;
            }

            println!("data {:#?}", csr_mat.indptr());
            println!("idxcount {:#?}", idxptrcount.as_slice());
            println!("idxdisplacements {:#?}", idxptrdisplacements.as_slice());

            let partition = mpi::datatype::Partition::new(
                csr_mat.indptr(),
                idxptrcount.as_slice(),
                idxptrdisplacements.as_slice(),
            );

            root_process.scatter_varcount_into_root(&partition, csr_indptr.as_mut_slice());

            // We also need to scatter the index pointers.
        } else {
            root_process.scatter_into(&mut my_data_count);

            let mut csr_data = vec![T::zero(); my_data_count];
            let mut csr_indices = vec![0 as usize; my_data_count];
            let mut csr_indptr = vec![0 as usize; 1 + my_number_of_rows];
            let shape = vec![0 as usize; 2];

            root_process.scatter_varcount_into(csr_data.as_mut_slice());

            root_process.scatter_varcount_into(csr_indices.as_mut_slice());
            root_process.scatter_varcount_into(csr_indptr.as_mut_slice());

            let first_elem = csr_indptr[0];
            csr_indptr
                .iter_mut()
                .for_each(|elem| *elem = *elem - first_elem);

            if my_rank == 1 {
                println!("Rank 2 data {:#?}", csr_data);
                println!("Rank 2 indices {:#?}", csr_indices);
                println!("Rank 2 ponter {:#?}", csr_indptr);
            }
        }
    }

    pub fn matmul(&self, alpha: T, x: &[T], beta: T, y: &mut [T]) {}

    // pub fn from_aij(
    //     shape: (IndexType, IndexType),
    //     rows: &[IndexType],
    //     cols: &[IndexType],
    //     data: &[T],
    // ) -> SparseLinAlgResult<Self> {
    //     let mut sorted: Vec<IndexType> = (0..rows.len()).collect();
    //     sorted.sort_by_key(|&idx| rows[idx]);

    //     let nelems = data.len();

    //     let mut indptr = Vec::<IndexType>::with_capacity(1 + shape.0);
    //     let mut indices = Vec::<IndexType>::with_capacity(nelems);
    //     let mut new_data = Vec::<T>::with_capacity(nelems);

    //     let mut count: IndexType = 0;

    //     for row in 0..(shape.0) {
    //         indptr.push(count);
    //         while count < nelems && row == rows[sorted[count]] {
    //             count += 1;
    //         }
    //     }
    //     indptr.push(count);

    //     for index in 0..nelems {
    //         indices.push(cols[sorted[index]]);
    //         new_data.push(data[sorted[index]]);
    //     }

    //     Ok(Self::new(shape, indices, indptr, new_data))
    // }
}

// #[cfg(test)]
// mod test {

//     use super::*;

//     #[test]
//     fn test_csr_from_aij() {
//         // Test the matrix [[1, 2], [3, 4]]
//         let rows = vec![0, 0, 1, 1];
//         let cols = vec![0, 1, 0, 1];
//         let data = vec![1.0, 2.0, 3.0, 4.0];

//         let csr = CsrMatrix::from_aij((2, 2), &rows, &cols, &data).unwrap();

//         assert_eq!(csr.data().len(), 4);
//         assert_eq!(csr.indices().len(), 4);
//         assert_eq!(csr.indptr().len(), 3);

//         //Test the matrix [[0, 0, 0], [2.0, 0, 0], [0, 0, 0]]
//         let rows = vec![1];
//         let cols = vec![0];
//         let data = vec![2.0];

//         let csr = CsrMatrix::from_aij((3, 3), &rows, &cols, &data).unwrap();

//         assert_eq!(csr.indptr()[0], 0);
//         assert_eq!(csr.indptr()[1], 0);
//         assert_eq!(csr.indptr()[2], 1);
//         assert_eq!(csr.indptr()[3], 1);
//     }

//     #[test]
//     fn test_csr_matmul() {
//         // Test the matrix [[1, 2], [3, 4]]
//         let rows = vec![0, 0, 1, 1];
//         let cols = vec![0, 1, 0, 1];
//         let data = vec![1.0, 2.0, 3.0, 4.0];

//         let csr = CsrMatrix::from_aij((2, 2), &rows, &cols, &data).unwrap();

//         // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
//         // Expected result is [35, 79].

//         let x = vec![3.0, 4.0];
//         let mut res = vec![1.0, 2.0];

//         csr.matmul(3.0, &x, 2.0, &mut res);

//         assert_eq!(res[0], 35.0);
//         assert_eq!(res[1], 79.0);
//     }
// }
