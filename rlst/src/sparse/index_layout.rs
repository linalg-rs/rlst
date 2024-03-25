//! Index layout
#[cfg(feature = "mpi")]
pub mod default_mpi_index_layout;

pub mod default_serial_index_layout;

#[cfg(feature = "mpi")]
pub use default_mpi_index_layout::*;

pub use default_serial_index_layout::*;
