//! Default implementations of operator concepts for RLST types.

pub mod array_vector_space;
#[cfg(feature = "mpi")]
pub mod distributed_array_vector_space;
// #[cfg(feature = "mpi")]
// pub mod distributed_sparse_operator;
// pub mod matrix_operator;

// pub use array_vector_space::{ArrayVectorSpace, ArrayVectorSpaceElement};
// #[cfg(feature = "mpi")]
// pub use distributed_array_vector_space::{
//     DistributedArrayVectorSpace, DistributedArrayVectorSpaceElement,
// };
// pub use matrix_operator::{MatrixOperator, MatrixOperatorRef};
