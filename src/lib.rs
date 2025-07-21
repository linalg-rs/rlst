//! The Rust linear solver toolbox (RLST).
//!
//! The purpose of this library is to provide a comprehensive set of tools
//! for dense and sparse linear algebra operations required in the solution
//! of partial differential equations and other problems.
//! RLST has the following feature set.
//! - n-dimensional array structures that can be allocated on the stack or heap.
//! - Support for BLAS matrix-matrix multiplication and a subset of Lapack operations (incl. LU, SVD, QR).
//! - CSR and CSC sparse matrices on a single node or via MPI on multiple nodes.
//! - An interface to UMFPACK for the solution of sparse linear systems.
//! - Import and export into Matrix-Market format.
//! - A general `operator` interface that can abstract linear operators, and iterative solvers
//!   acting on linear operators.
//!
//! To learn about the features of RLST please have a look at the following documents.
//! - [Initialising RLST.](crate::doc::initialise_rlst)
//! - [An introduction to dense linear algebra with RLST](crate::doc::dense_linear_algebra).
//! - Sparse matrix operations.
//! - Using the operator interface for iterative solvers.
//! - Import and export of matrices.
//! - BLAS dependencies, multithreading, and GPU offloading.

#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod base_types;
pub mod dense;
pub mod distributed_tools;
pub mod simd;
pub mod sparse;
pub mod traits;
// pub mod doc;
// pub mod external;
pub mod io;
// pub mod tracing;
//pub mod distributed_tools;

//pub mod prelude;
// pub mod threading;

// pub mod operator;

//pub use prelude::*;

// Re-exports
pub use rlst_proc_macro::rlst_dynamic_array;
pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use crate::dense::array::empty_array;
pub use crate::dense::array::Array;

// Re-export the traits
pub use traits::*;

// Re-export the base types
pub use base_types::*;

#[cfg(test)]
mod test {
    use criterion as _; // Hack to show that criterion is used, as cargo test does not see benches
}
