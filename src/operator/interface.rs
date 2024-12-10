//! Default implementations of operator concepts for RLST types.

pub mod array_vector_space;
pub mod dense_matrix_operator;
#[cfg(feature = "mpi")]
pub mod distributed_array_vector_space;
#[cfg(feature = "mpi")]
pub mod distributed_sparse_operator;
pub mod sparse_operator;

pub use array_vector_space::{ArrayVectorSpace, ArrayVectorSpaceElement};
pub use dense_matrix_operator::DenseMatrixOperator;
#[cfg(feature = "mpi")]
pub use distributed_array_vector_space::{
    DistributedArrayVectorSpace, DistributedArrayVectorSpaceElement,
};
pub use sparse_operator::CscMatrixOperator;
pub use sparse_operator::CsrMatrixOperator;
