//! The Rust linear solver toolbox (RLST).
//!
//! The purpose of this library is to provide a comprehensive set of tools
//! for dense and sparse linear algebra operations required in the solution
//! of partial differential equations and other problems.
//! RLST has the following feature set.
//! - n-dimensional array structures that can be allocated on the stack or heap.
//! - Support for BLAS matrix-matrix multiplication and a subset of Lapack operations (incl. LU, SVD, QR).
//! - CSR sparse matrices on a single node or via MPI on multiple nodes.
//! - Import and export into Matrix-Market format.
//! - A general `operator` interface that can abstract linear operators, and iterative solvers
//!   acting on linear operators.
//!
//! To learn about the features of RLST please have a look at the following documents.
//! - [Getting started with RLST](crate::doc::getting_started)
//! - [An introduction to dense linear algebra with RLST](crate::doc::dense_linear_algebra)
//! - [Matrix decompositions](crate::doc::matrix_decompositions)
//! - [Sparse matrix operations](crate::doc::sparse_matrices)
//! - [Abstract linear algebra and iterative solvers](crate::doc::abstract_linear_algebra)
//! - [MPI distributed computations](crate::doc::distributed_computations)

#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod base_types;
pub mod dense;
#[cfg(feature = "mpi")]
pub mod distributed_tools;
pub mod doc;
pub mod io;
pub mod simd;
pub mod sparse;
pub mod tracing;
pub mod traits;

pub mod threading;

pub mod operator;

// Re-exports
pub use rlst_proc_macro::measure_duration;
pub use rlst_proc_macro::rlst_dynamic_array;
pub use rlst_proc_macro::rlst_static_array;
pub use rlst_proc_macro::rlst_static_type;

pub use crate::dense::array::Array;
pub use crate::dense::array::empty_array;

// Re-export the traits
pub use traits::*;

// Re-export the base types
pub use base_types::*;

pub use dense::array::DynArray;
pub use dense::array::SliceArray;
pub use dense::array::SliceArrayMut;
pub use dense::array::StridedDynArray;

// Important constants

/// Align memory along cache lines
pub const CACHE_ALIGNED: usize = aligned_vec::CACHELINE_ALIGN;

/// Align memory for FFTW. FFTW requires 16 bytes alignment
pub const FFTW_ALIGNED: usize = 16;

/// Align memory along page lines
///
// On x86_64 architectures a page size of 4kb is assumed.
// On other architectures we default to 64kb to take into account
// large page sizes on some ARM platforms.
pub const PAGE_ALIGNED: usize = {
    if cfg!(target_arch = "x86_64") {
        4096
    } else {
        65536
    }
};

#[cfg(test)]
mod tests {

    extern crate blas_src;
    extern crate lapack_src;
}
