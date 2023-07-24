//! Basic types

pub use cauchy::{c32, c64, Scalar};
use thiserror::Error;

/// The RLST error type.
#[derive(Error, Debug)]
pub enum RlstError {
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError { expected: usize, actual: usize },
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    #[error("Lapack error: {0}")]
    LapackError(i32),
    #[error("{0}")]
    GeneralError(String),
    #[error("I/O Error: {0}")]
    IoError(String),
    #[error("Umfpack Error Code: {0}")]
    UmfpackError(i32),
}

/// Alias for an RLST Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;
