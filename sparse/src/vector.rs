#[cfg(feature = "mpi")]
pub mod default_mpi_vector;
pub mod default_serial_vector;

#[cfg(feature = "mpi")]
pub use default_mpi_vector::*;
pub use default_serial_vector::*;
