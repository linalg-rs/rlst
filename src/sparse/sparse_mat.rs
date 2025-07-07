//! Sparse matrices
//pub mod csc_mat;
pub mod csr_mat;

#[cfg(feature = "mpi")]
pub mod distributed_csr_mat;
#[cfg(feature = "suitesparse")]
pub mod umfpack;
