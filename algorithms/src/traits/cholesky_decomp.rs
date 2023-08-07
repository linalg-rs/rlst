//! Trait for Cholesky decomposition.
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{Matrix, MatrixD, MatrixImplTrait, SizeIdentifier};

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
    fn solve<MatImpl: MatrixImplTrait<Self::T, RS, CS>, RS: SizeIdentifier, CS: SizeIdentifier>(
        &self,
        rhs: &Matrix<Self::T, MatImpl, RS, CS>,
    ) -> RlstResult<Self::Sol>;
}

/// Return the Cholesky decomposition of a Hermitican positive definite matrix.
pub trait Cholesky {
    type T: Scalar;
    type Out;

    fn cholesky(self) -> RlstResult<Self::Out>;
}
