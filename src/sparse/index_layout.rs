//! Index layout
#[cfg(feature = "mpi")]
pub mod default_distributed_index_layout;

#[cfg(feature = "mpi")]
pub use default_distributed_index_layout::*;
