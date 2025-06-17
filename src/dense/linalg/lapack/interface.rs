//! LAPACK interface for RLST.

use ev::Ev;
use gees::Gees;
use geev::Geev;
use gels::Gels;
use geqp3::Geqp3;
use geqrf::Geqrf;
use gesdd::Gesdd;
use gesvd::Gesvd;
use getrf::Getrf;
use getri::Getri;
use getrs::Getrs;
use mqr::Mqr;
pub use num::complex::Complex32 as c32;
pub use num::complex::Complex64 as c64;
use orgqr::Orgqr;
use posv::Posv;
use potrf::Potrf;
use thiserror::Error;

use crate::RlstScalar;

pub mod ev;
pub mod gees;
pub mod geev;
pub mod gels;
pub mod geqp3;
pub mod geqrf;
pub mod gesdd;
pub mod gesvd;
pub mod getrf;
pub mod getri;
pub mod getrs;
pub mod mqr;
pub mod orgqr;
pub mod posv;
pub mod potrf;

/// Basic dimension type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim(usize, usize);

impl From<(usize, usize)> for Dim {
    fn from(dim: (usize, usize)) -> Self {
        Dim(dim.0, dim.1)
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

/// The Rlst error type.
#[derive(Error, Debug)]
pub enum LapackError {
    /// Size mismatch
    #[error("Dimension mismatch. Expected {expected}. Actual {actual}.")]
    DimensionMismatch {
        /// Expected dimension
        expected: Dim,
        /// Actual dimension
        actual: Dim,
    },
    /// Length mismatch
    #[error("Length mismatch. Expected {expected}. Actual {actual}.")]
    LengthMismatch {
        /// Expected length
        expected: usize,
        /// Actual length
        actual: usize,
    },
    /// Info code from LAPACK
    #[error("LAPACK error code: {0}")]
    LapackInfoCode(i32),
}

/// Alias for a Lapack Result type.
pub type LapackResult<T> = std::result::Result<T, LapackError>;

/// Helper function to convert LAPACK info codes to a `LapackResult`.
pub(crate) fn lapack_return<T>(info: i32, result: T) -> LapackResult<T> {
    if info == 0 {
        Ok(result)
    } else {
        Err(LapackError::LapackInfoCode(info))
    }
}

/// Trait for types that implement LAPACK functionality.
pub trait Lapack:
    Ev
    + Gees
    + Geev
    + Gels
    + Geqp3
    + Geqrf
    + Gesdd
    + Gesvd
    + Getrf
    + Getri
    + Getrs
    + Mqr
    + Orgqr
    + Potrf
    + Posv
    + RlstScalar
{
}

impl<T> Lapack for T where
    T: Ev
        + Gees
        + Geev
        + Gels
        + Geqp3
        + Geqrf
        + Gesdd
        + Gesvd
        + Getrf
        + Getri
        + Getrs
        + Mqr
        + Orgqr
        + Potrf
        + Posv
        + RlstScalar
{
}
