//! Default implementations of operator concepts for RLST types.

pub mod array_vector_space;
#[cfg(feature = "mpi")]
pub mod distributed_array_vector_space;
// #[cfg(feature = "mpi")]
// pub mod distributed_sparse_operator;

pub mod matrix_operator;
