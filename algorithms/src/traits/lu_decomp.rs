//! Trait for LU Decomposition and linear system solves with LU.
use crate::lapack::TransposeMode;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{DataContainerMut, GenericBaseMatrix, MatrixD, SizeIdentifier};

/// Defines the LU Decomposition of a matrix.
///
/// The LU Decomposition for a general `(m, n)` matrix has the form
/// `PA = LU` with `P` a `(m, m)` permutation matrix, `L`, a `(m, k)` lower triangular (or trapezoidal)
/// matrix with ones on the diagonal, and `U` a `(k, n)` upper triangular (or trapezoidal) matrix.
/// Here, `k=min(m, n)`. The [`LUDecomp::get_l`] and [`LUDecomp::get_u`] routines can be used to obtain the `L` and `U` matrices.
/// The `P` matrix is implicitly defined through a permutation vector `perm` returned by [`LUDecomp::get_perm`].
/// It is defined in such a way that the ith row of `LU` is identical to row `perm[i]` of the original
/// matrix `A`. To solve a linear system `Ax=b` the routine [`LUDecomp::solve`] is provided.
pub trait LUDecomp {
    type T: Scalar;

    /// Raw pointer to the data.
    fn data(&self) -> &[Self::T];

    /// Return the L matrix.
    fn get_l(&self) -> MatrixD<Self::T>;

    /// Return the U matrix.
    fn get_u(&self) -> MatrixD<Self::T>;

    /// Return the permutation vector.
    fn get_perm(&self) -> Vec<usize>;

    /// Return the shape of the original matrix.
    fn shape(&self) -> (usize, usize);

    /// Solve for a right-hand side.
    fn solve<Data: DataContainerMut<Item = Self::T>, RhsR: SizeIdentifier, RhsC: SizeIdentifier>(
        &self,
        rhs: &mut GenericBaseMatrix<Self::T, Data, RhsR, RhsC>,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}
