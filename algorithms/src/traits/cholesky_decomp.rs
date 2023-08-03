//! Trait for Cholesky decomposition.
use super::types::*;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{MatrixD, RandomAccessByValue, Shape};

/// Defines the Cholesky decomposition of a Hermitian positive definite matrix.
///
/// The Cholesky decomposition of a Hermitian positive definite matrix `A` is defined as
/// `A=U^HU`, where `U` is upper triangular and `U^H` denotes the complex conjugate transpose
/// of `U`. Alternatively, the Cholesky decomposition can be defined as `A=L^H` for `L` lower
/// triangular.
pub trait CholeskyDecomp {
    type T: Scalar;
    type Sol;

    /// Raw pointer to the data.
    fn data(&self) -> &[Self::T];

    /// Return the L matrix.
    fn get_l(&self) -> MatrixD<Self::T>;

    /// Return the U matrix.
    fn get_u(&self) -> MatrixD<Self::T>;

    /// Return the shape of the original matrix.
    fn shape(&self) -> (usize, usize);

    /// Solve for a right-hand side.
    fn solve<Rhs: RandomAccessByValue<Item = Self::T> + Shape>(
        &self,
        rhs: &Rhs,
    ) -> RlstResult<Self::Sol>;
}

/// Return the Cholesky decomposition of a Hermitican positive definite matrix.
/// If `triangular_type` is [`TriangularType::Upper`] then only the upper triangular
/// part of the matrix is used. If `triangular_type` is [`TriangularType::Lower`] then
/// only the lower triangular part is used.
pub trait Cholesky {
    type T: Scalar;
    type Out;

    fn cholesky(self, triangular_type: TriangularType) -> RlstResult<Self::Out>;
}
