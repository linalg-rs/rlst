//! Traits for linear algebra operations on matrices.

use std::collections::btree_set::SymmetricDifference;

use crate::{
    dense::{
        array::{Array, DynArray},
        types::{RlstResult, TransMode},
    },
    BaseItem, RlstScalar,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix.
    Upper,
    /// Lower triangular matrix.
    Lower,
}

use super::lapack::{
    interface::Lapack,
    lu::LuDecomposition,
    qr::{EnablePivoting, QrDecomposition},
    symmeig::SymmEigMode,
};

/// Compute the matrix inverse.
pub trait Inverse {
    /// The item type of the inverse.
    type Output;

    /// Compute the inverse of a matrix.
    fn inverse(&self) -> RlstResult<Self::Output>;
}

/// Compute the LU decomposition of a matrix.
///
/// The LU decomposition is defined as `A = P * L * U`, where:
/// - `A` is the original matrix,
/// - `P` is a permutation matrix,
/// - `L` is a lower triangular matrix with unit diagonal,
/// - `U` is an upper triangular matrix.
pub trait Lu {
    /// Item type of the LU decomposition.
    type Item: Lapack;
    /// Compute the LU decomposition of a matrix.
    fn lu(&self) -> RlstResult<LuDecomposition<Self::Item>>;
}

/// Compute the QR decomposition of a matrix.
pub trait Qr {
    /// Item type of the QR decomposition.
    type Item: Lapack;

    /// Compute the QR decomposition of a matrix.
    fn qr(&self, pivoting: EnablePivoting) -> RlstResult<QrDecomposition<Self::Item>>;
}

/// Compute the symmetric eigenvalue decomposition of a matrix.
pub trait SymmEig {
    /// Item type of the symmetric eigenvalue decomposition.
    type Item: Lapack;

    /// Compute the symmetric eigenvalue decomposition of a matrix.
    fn symm_eig(
        &self,
        uplo: UpLo,
        mode: SymmEigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        Option<DynArray<Self::Item, 2>>,
    )>;
}
