//! Traits for matrix decompositions and solving linear systems.

use num::Zero;

use itertools::Itertools;

use crate::{
    base_types::{RlstResult, UpLo},
    dense::{
        array::DynArray,
        linalg::lapack::{
            eigenvalue_decomposition::EigMode,
            lu::LuDecomposition,
            pseudo_inverse::PInv,
            qr::{EnablePivoting, QrDecomposition},
            singular_value_decomposition::SvdMode,
            symmeig::SymmEigMode,
        },
    },
    traits::{
        base_operations::{ConjObject, EvaluateObject, Len, Shape, ToType},
        iterators::ArrayIteratorByValue,
        rlst_num::RlstScalar,
    },
    AsOwnedRefType,
};

use super::{base::Gemm, lapack::Lapack};

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
    #[allow(clippy::type_complexity)]
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
    ///
    /// and encode the eigenvalues of the matrix.
    /// - A unitary matrix `Z` such that `A = Z * T * Z^H`, where `Z^H` is the conjugate transpose
    #[allow(clippy::type_complexity)]
    fn schur(&self) -> RlstResult<(DynArray<Self::Item, 2>, DynArray<Self::Item, 2>)>;

    /// Compute the eigenvalues and eigenvectors of the matrix.
    ///
    /// The function returns a tuple `(lam, v, w)` containing:
    /// - A vector `lam` of eigenvalues.
    /// - An optional matrix `v` of right eigenvectors.
    /// - An optional matrix `w` of left eigenvectors.
    #[allow(clippy::type_complexity)]
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
    type Item: Lapack + Gemm;

    /// Compute the singular values of a matrix.
    fn singularvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>>;

    /// Compute the singular value decomposition of a matrix.
    ///
    /// The function returns a tuple containing:
    /// - A vector of singular values.
    /// - A matrix `U` containing the left singular vectors.
    /// - A matrix `Vh` containing the right singular vectors as rows.
    #[allow(clippy::type_complexity)]
    fn svd(
        &self,
        mode: SvdMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )>;

    /// Compute the truncated singular value decomposition of a matrix.
    ///
    /// **Arguments:**
    /// - `max_singular_values`: Maximum number of singular values to compute. If `None`, all
    ///   singular values are computed.
    /// - `tol`: Relative tolerance for truncation. Singular values smaller or equal to `tol *
    ///   s[0]`, where `s[0]` is the largest singular value, will be discarded. Zero singular values
    ///   are always discarded.
    ///
    /// Returns a tuple containing:
    /// - A vector of singular values
    /// - A matrix `U` containing the left singular vectors.
    /// - A matrix `Vh` containing the right singular vectors as rows.
    #[allow(clippy::type_complexity)]
    fn svd_truncated(
        &self,
        max_singular_values: Option<usize>,
        tol: Option<<Self::Item as RlstScalar>::Real>,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )> {
        let (s, u, vh) = self.svd(SvdMode::Compact)?;

        let nvalues = std::cmp::min(
            match max_singular_values {
                Some(n) => n,
                None => s.len(),
            },
            s.len(),
        );

        let tol = match tol {
            Some(t) => t,
            None => <<Self::Item as RlstScalar>::Real as Zero>::zero(),
        };

        let count = match s
            .iter_value()
            .take(nvalues)
            .find_position(|&elem| elem <= tol * s[[0]])
        {
            Some((index, _)) => index,
            None => nvalues,
        };

        let s = s.into_subview([0], [count]).eval();
        let u = u.r().into_subview([0, 0], [u.shape()[0], count]).eval();
        let vh = vh.r().into_subview([0, 0], [count, vh.shape()[1]]).eval();

        Ok((s, u, vh))
    }

    /// Compute the pseudo-inverse of a matrix.
    fn pseudo_inverse(
        &self,
        max_singular_values: Option<usize>,
        tol: Option<<Self::Item as RlstScalar>::Real>,
    ) -> RlstResult<PInv<Self::Item>>
    where
        <Self::Item as RlstScalar>::Real: Into<Self::Item>,
    {
        let (s, u, vh) = self.svd_truncated(max_singular_values, tol)?;

        Ok(PInv::new(
            s.into_type().eval(),
            u.conj().transpose().eval(),
            vh.conj().transpose().eval(),
        ))
    }
}

/// Generic trait for solving square or rectangular linear systems.
pub trait Solve<Rhs> {
    /// The output type of the solver.
    type Output;

    /// Solve the linear system `Ax = b` for `x`.
    // If `A` is not square, the system is solved in the least-squares sense.
    fn solve(&self, rhs: &Rhs) -> RlstResult<Self::Output>;
}

/// Cholesky decomposition for positive definite matrices.
pub trait Cholesky {
    /// Item type of the array.
    type Item;

    /// Compute the Cholesky decomposition of a positive definite matrix.
    ///
    /// **Arguments:**
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    fn cholesky(&self, uplo: UpLo) -> RlstResult<DynArray<Self::Item, 2>>;
}

/// Cholesky solver for positive definite systems.
pub trait CholeskySolve<Rhs> {
    /// The output type of the Cholesky solver.
    type Output;

    /// Solve a positive definite system of linear equations using Cholesky factorization.
    ///
    /// **Arguments:**
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    fn cholesky_solve(&self, uplo: UpLo, rhs: &Rhs) -> RlstResult<Self::Output>;
}

/// Solve a triangular system of linear equations.
pub trait SolveTriangular<Rhs> {
    /// The output type of the triangular solver.
    type Output;

    /// Solve a triangular system of linear equations.
    ///
    /// **Arguments:**
    /// - `uplo`: Specifies whether the upper or lower triangular part of the matrix is stored.
    fn solve_triangular(&self, uplo: UpLo, rhs: &Rhs) -> RlstResult<Self::Output>;
}
