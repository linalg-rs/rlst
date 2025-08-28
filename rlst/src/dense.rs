//! Dense array types and operations on them.
//!
pub mod base_array;
pub mod data_container;
pub mod linalg;
pub mod strided_base_array;
pub mod tools;

pub mod array;
pub mod gemm;
pub mod layout;
pub mod macros;
pub mod matrix_multiply;

#[cfg(feature = "mpi")]
pub mod block_cyclic_array;
