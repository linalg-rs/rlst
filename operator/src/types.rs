//! Basic types

// The scalar type used in the library.
pub use cauchy::Scalar;
use thiserror::Error;

// The `IndexType` is used whenever we use an integer counting type.
//
// By default it should be `usize`.
pub type IndexType = usize;

#[derive(Error, Debug)]
pub enum SparseLinAlgError {
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError {
        expected: IndexType,
        actual: IndexType,
    },
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
}

pub type SparseLinAlgResult<T> = std::result::Result<T, SparseLinAlgError>;
