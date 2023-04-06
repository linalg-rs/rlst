//! Definition of CSR matrices.

use crate::ghost_communicator::GhostCommunicator;
use crate::index_layout::DefaultMpiIndexLayout;
use crate::sparse::csr_mat::CsrMatrix;
use crate::sparse::SparseMatType;
use crate::traits::index_layout::IndexLayout;
use mpi::datatype::{Partition, PartitionMut};
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};

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
