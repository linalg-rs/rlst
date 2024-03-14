//! Default implementations of operator concepts for RLST types.

pub mod array_vector_space;
pub mod dense_matrix_operator;
pub mod sparse_operator;

pub use array_vector_space::{ArrayVectorSpace, ArrayVectorSpaceElement};
pub use dense_matrix_operator::DenseMatrixOperator;
pub use sparse_operator::CscMatrixOperator;
pub use sparse_operator::CsrMatrixOperator;
