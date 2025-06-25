//! Traits for linear algebra operations on matrices.

use crate::{
    dense::{array::DynArray, types::RlstResult},
    RlstScalar,
};

/// Determine whether a matrix is upper or lower triangular.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpLo {
    /// Upper triangular matrix.
    Upper,
    /// Lower triangular matrix.
    Lower,
}

use super::lapack::{
    eigenvalue_decomposition::EigMode,
    interface::Lapack,
    lu::LuDecomposition,
    qr::{EnablePivoting, QrDecomposition},
    singular_value_decomposition::SvdMode,
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

    /// Compute the eigenvalues of a real symmetric or complex Hermitian matrix.
    fn eigenvaluesh(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>> {
        Ok(self.eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly)?.0)
    }

    /// Compute the symmetric eigenvalue decomposition of a matrix.
    fn eigh(
        &self,
        uplo: UpLo,
        mode: SymmEigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        Option<DynArray<Self::Item, 2>>,
    )>;
}

/// Compute the eigenvalue decomposition of a matrix.
pub trait EigenvalueDecomposition {
    /// The item type of the matrix.
    type Item: Lapack;

    /// Return the eigenvalues of the matrix.
    fn eigenvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Complex, 1>>;

    /// Compute the Schur decomposition of the matrix.
    /// Returns a tuple containing:
    /// - A block-upper triangular matrix `T`. The diagonal blocks are 1x1 or 2x2.
    /// and encode the eigenvalues of the matrix.
    /// - A unitary matrix `Z` such that `A = Z * T * Z^H`, where `Z^H` is the conjugate transpose
    fn schur(&self) -> RlstResult<(DynArray<Self::Item, 2>, DynArray<Self::Item, 2>)>;

    /// Compute the eigenvalues and eigenvectors of the matrix.
    ///
    /// The function returns a tuple `(lam, v, w)` containing:
    /// - A vector `lam` of eigenvalues.
    /// - An optional matrix `v` of right eigenvectors.
    /// - An optional matrix `w` of left eigenvectors.
    fn eig(
        &self,
        mode: EigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Complex, 1>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
    )>;
}

/// Compute the singular value decomposition of a matrix.
pub trait SingularvalueDecomposition {
    /// The item type of the matrix.
    type Item: Lapack;

    /// Compute the singular values of a matrix.
    fn singularvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>>;

    /// Compute the singular value decomposition of a matrix.
    ///
    /// The function returns a tuple containing:
    /// - A vector of singular values.
    /// - A matrix `U` containing the left singular vectors.
    /// - A matrix `Vh` containing the right singular vectors as rows.
    fn svd(
        &self,
        mode: SvdMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )>;
}
