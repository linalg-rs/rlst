//! Sparse matrix traits

use crate::sparse::SparseMatType;

use super::BaseItem;

/// Return the number of non-zero entries in the sparse matrix.
pub trait Nonzeros {
    /// Return the number of non-zero entries.
    fn nnz(&self) -> usize;
}

/// Return the type of the sparse matrix.
pub trait SparseMatrixType {
    /// Return the type of the sparse matrix.
    fn mat_type(&self) -> SparseMatType;
}

/// Construct a sparse matrix from an iterator of (i, j, value) tuples.
pub trait FromAijIterator<I: Iterator<Item = ([usize; 2], Self::Item)>>
where
    Self: BaseItem,
{
    /// Create a sparse matrix from an iterator of (i, j, value) tuples.
    fn from_aij_iter(iter: I, shape: [usize; 2]) -> Self;
}
