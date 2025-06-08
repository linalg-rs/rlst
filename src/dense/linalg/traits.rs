//! Traits for linear algebra operations on matrices.

use crate::{
    dense::{
        array::{Array, DynArray},
        types::{RlstResult, TransMode},
    },
    BaseItem,
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
    /// The output type of the LU decomposition.
    type Output;

    /// Compute the LU decomposition of a matrix.
    fn lu(&self) -> RlstResult<Self::Output>;
}

/// Solve a linear system of equations `Ax = b` for a matrix `A` and right-hand side `b`.
pub trait Solve<Rhs> {
    /// The output type of the solver.
    type Output;

    /// Solve the linear system `Ax = b` for a vector `b`.
    fn solve(&self, trans: TransMode, b: &Rhs) -> RlstResult<Self::Output>;
}

/// Get the `L` matrix from the LU decomposition.
pub trait GetL {
    /// The output type.
    type Output;

    /// Get the left-hand side matrix `L` from the LU decomposition.
    fn l_mat(&self) -> RlstResult<Self::Output>;
}

/// Get the `U` matrix from the LU decomposition.
pub trait GetU {
    /// The output type.
    type Output;

    /// Get the right-hand side matrix `U` from the LU decomposition.
    fn u_mat(&self) -> RlstResult<Self::Output>;
}

/// Get the permutation matrix P from the LU decomposition.
pub trait GetP {
    /// The output type.
    type Output;

    /// Get the permutation matrix `P` from the LU decomposition.
    fn p_mat(&self) -> RlstResult<Self::Output>;
}

/// Get both `L` and `U` matrices from the LU decomposition.
pub trait GetLU {
    /// The output type.
    type Output;

    /// Get the left-hand side matrix `L` and right-hand side matrix `U` from the LU decomposition.
    fn l_u_mat(&self) -> RlstResult<Self::Output>;
}

/// Get the permutation vector from the LU decomposition.
///
/// If `perm[i] = j`, then the `i`-th row of the LU decomposition corresponds to the `j`-th row of
/// the original matrix.
pub trait GetPermVec {
    /// The output type.

    /// Get the permutation vector from the LU decomposition.
    fn perm_vec(&self) -> RlstResult<Vec<usize>>;
}

/// Get the determinant of the matrix from the LU decomposition.
pub trait GetDeterminant {
    /// The item type.
    type Item;

    /// Compute the determinant of the matrix from the LU decomposition.
    fn det(&self) -> Self::Item;
}

/// Compute the QR decomposition of a matrix.
pub trait Qr {
    /// The output type of the QR decomposition.
    type Output;

    /// Compute the QR decomposition of a matrix.
    fn qr(&self) -> RlstResult<Self::Output>;
}

/// Solve a least squares problem.
pub trait QrSolve<Rhs> {
    /// The output type of the solver.
    type Output;

    /// Solve the least squares problem with the QR decomposition.
    fn qr_solve(&self);
}

/// Return the `Q` matrix from the QR decomposition.
pub trait GetQ {
    /// The output type.
    type Output;

    /// Get the orthogonal matrix `Q` from the QR decomposition.
    fn q_mat(&self) -> RlstResult<Self::Output>;
}

/// Return the `R` matrix from the QR decomposition.
pub trait GetR {
    /// The output type.
    type Output;

    /// Get the upper triangular matrix `R` from the QR decomposition.
    fn r_mat(&self) -> RlstResult<Self::Output>;
}

/// Return the permutation vector from a pivoted QR decomposition.
pub trait GetQRPivot {
    /// Get the permutation vector from the QR decomposition.
    fn qr_pivot(&self) -> RlstResult<Vec<usize>>;
}
