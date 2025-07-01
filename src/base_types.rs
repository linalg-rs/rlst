//! Basic types.

mod number_traits_impl;
mod rlst_num;
mod rlst_num_impl;

pub use rlst_num::*;

use thiserror::Error;

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum RlstError {
    /// Not implemented
    #[error("Method {0} is not implemented.")]
    NotImplemented(String),
    /// Operation failed
    #[error("Operation {0} failed.")]
    OperationFailed(String),
    /// Matrix is empty
    #[error("Matrix has empty dimension {0:#?}.")]
    MatrixIsEmpty((usize, usize)),
    /// Dimension mismatch
    #[error("Dimension mismatch. Expected {expected:}. Actual {actual:}")]
    SingleDimensionError {
        /// Expected dimension
        expected: usize,
        /// Actual dimension
        actual: usize,
    },
    /// Index layout error
    #[error("Index Layout error: {0}")]
    IndexLayoutError(String),
    /// MPI rank error
    #[error("MPI Rank does not exist. {0}")]
    MpiRankError(i32),
    /// Incompatible stride for Lapack
    #[error("Incompatible stride for Lapack.")]
    IncompatibleStride,
    /// Lapack error
    #[error("Lapack error: {0}")]
    LapackError(#[from] LapackError),
    /// General error
    #[error("{0}")]
    GeneralError(String),
    /// I/O error
    #[error("I/O Error: {0}")]
    IoError(String),
    /// Umfpack error
    #[error("Umfpack Error Code: {0}")]
    UmfpackError(i32),
    /// Matrix is not square
    #[error("Matrix is not square. Dimension: {0}x{1}")]
    MatrixNotSquare(usize, usize),
    /// Matrix is not Hermitian
    #[error("Matrix is not Hermitian (complex conjugate symmetric).")]
    MatrixNotHermitian,
}

/// Alias for an Rlst Result type.
pub type RlstResult<T> = std::result::Result<T, RlstError>;

/// Transposition Mode.
#[derive(Clone, Copy, PartialEq)]
pub enum TransMode {
    /// No modification of matrix.
    NoTrans,
    /// Complex conjugate of matrix.
    ConjNoTrans,
    /// Transposition of matrix.
    Trans,
    /// Conjugate transpose of matrix.
    ConjTrans,
}

// The following is a workaround to allow for implied trait bounds on associated types.
// A full descripton can be found at: https://docs.rs/imply-hack/latest/imply_hack/index.html

/// Trait to specify associated bounds as super trait bounds.
/// This allows the bounds to be implied and not explicitly stated in the type signature.
/// See the [imply-hack](https://docs.rs/imply-hack/latest/imply_hack/index.html) crate for more
/// details.
pub trait Imply<T>: sealed::ImplyInner<T, Is = T> {}

impl<T, U> Imply<T> for U {}

mod sealed {
    pub trait ImplyInner<T> {
        type Is;
    }

    impl<T, U> ImplyInner<T> for U {
        type Is = T;
    }
}

/// Memory layout of an object
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Column major
    ColumnMajor,
    /// Row major
    RowMajor,
    /// Unknown
    Unknown,
}

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum LapackError {
    /// Info code from LAPACK
    #[error("LAPACK error code: {0}")]
    LapackInfoCode(i32),
}

/// Alias for a Lapack Result type.
pub type LapackResult<T> = std::result::Result<T, LapackError>;
