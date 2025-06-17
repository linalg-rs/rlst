//! Traits for linear algebra operations on matrices.

use crate::{
    dense::{
        array::{Array, DynArray},
        types::{RlstResult, TransMode},
    },
    BaseItem,
};

use super::lapack::{
    interface::Lapack,
    lu::LuDecomposition,
    qr::{EnablePivoting, QrDecomposition},
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
