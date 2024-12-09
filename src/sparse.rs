//! RLST sparse
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

#[cfg(feature = "mpi")]
pub mod distributed_vector;

pub mod index_layout;
pub mod sparse_mat;
pub mod tools;
pub mod traits;
