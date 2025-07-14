//! Sparse matrix traits

/// Return the number of non-zero entries in the sparse matrix.
pub trait Nonzeros {
    /// Return the number of non-zero entries.
    fn nnz(&self) -> usize;
}
