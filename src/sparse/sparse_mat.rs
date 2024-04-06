//! Sparse matrices
pub mod csc_mat;
pub mod csr_mat;

#[cfg(feature = "mpi")]
pub mod mpi_csr_mat;

pub mod tools;
#[cfg(feature = "suitesparse")]
pub mod umfpack;

/// Sparse matrix type
#[derive(Copy, Clone)]
pub enum SparseMatType {
    /// CSR matrix
    Csr,
    /// CSC matrix
    Csc,
}
