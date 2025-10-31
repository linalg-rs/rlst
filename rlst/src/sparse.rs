//! RLST sparse
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

//pub mod sparse_mat;
pub mod binary_operator;
pub mod csr_mat;
#[cfg(feature = "mpi")]
pub mod distributed_csr_mat;
pub mod mat_operations;
#[cfg(feature = "mpi")]
pub mod mat_operations_distributed;
pub mod tools;
pub mod unary_aij_operator;

#[cfg(feature = "mpi")]
pub mod distributed_array;

/// Sparse matrix type
#[derive(Debug, Copy, Clone)]
pub enum SparseMatType {
    /// CSR matrix
    Csr,
    /// CSC matrix
    Csc,
    /// Distributed CSR matrix
    DistCsr,
}
