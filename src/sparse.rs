//! RLST sparse
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

//pub mod sparse_mat;
pub mod tools;

#[cfg(feature = "mpi")]
pub mod distributed_array;

/// Sparse matrix type
#[derive(Copy, Clone)]
pub enum SparseMatType {
    /// CSR matrix
    Csr,
    /// CSC matrix
    Csc,
}
