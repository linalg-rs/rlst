//! RLST sparse
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

#[cfg(feature = "mpi")]
pub mod distributed_vector;

#[cfg(feature = "mpi")]
pub mod ghost_communicator;

pub mod index_layout;
pub mod sparse;
pub mod tools;
pub mod traits;
