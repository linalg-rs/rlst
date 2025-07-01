//! LAPACK interface for RLST.

use crate::{
    base_types::{LapackError, LapackResult, RlstScalar},
    dense::linalg::lapack::interface::{
        ev::Ev, gees::Gees, geev::Geev, gels::Gels, geqp3::Geqp3, geqrf::Geqrf, gesdd::Gesdd,
        gesvd::Gesvd, getrf::Getrf, getri::Getri, getrs::Getrs, mqr::Mqr, orgqr::Orgqr, posv::Posv,
        potrf::Potrf, trsm::Trsm,
    },
};

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
    + Trsm
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
        + Trsm
        + RlstScalar
{
}
