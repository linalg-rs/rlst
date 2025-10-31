//! The RLST documentation.

pub mod abstract_linear_algebra;
pub mod dense_linear_algebra;
#[cfg(feature = "mpi")]
pub mod distributed_computations;

#[cfg(not(feature = "mpi"))]
pub mod distributed_computations {

    //! MPI not enabled. To enable MPI and show the corresponding documentation
    //! please recompile with the feature flag `mpi` enabled.
}

pub mod getting_started;
pub mod matrix_decompositions;
pub mod sparse_matrices;
