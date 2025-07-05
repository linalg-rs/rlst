//! LAPACK interface for RLST.

use crate::base_types::LapackError;
use crate::base_types::LapackResult;

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
pub mod trsm;

/// Helper function to convert LAPACK info codes to a `LapackResult`.
pub(crate) fn lapack_return<T>(info: i32, result: T) -> LapackResult<T> {
    if info == 0 {
        Ok(result)
    } else {
        Err(LapackError::LapackInfoCode(info))
    }
}
