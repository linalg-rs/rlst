//! Sparse matrix traits

use crate::{dense::array::DynArray, distributed_tools::IndexLayout, sparse::SparseMatType};

use super::{AijIteratorByValue, BaseItem, Shape};

/// Traits for sparse matrices.
pub trait SparseMat: BaseItem + Shape<2> {
    /// The type of the index set.
    type IndexSet;

    /// An iterator that iterates over non-zero elements of the sparse matrix.
    type AijIter: AijIteratorByValue<Item = Self::Item>;

    /// A mutable iterator that iterates over non-zero elements of the sparse matrix.
    type AijIterMut: AijIteratorByValue<Item = Self::Item>;

    /// Get the number of non-zero elements
    fn nnz(&self) -> usize;

    /// Get the type of the sparse matrix
    fn sparse_mat_type(&self) -> SparseMatType;

    /// Return an iterator over the local non-zero elements in the form (row, column, value).
    fn iter(&self) -> Self::AijIter;

    /// Return a mutable iterator over the local non-zero elements in the form (row, column,
    /// value).
    fn iter_mut(&mut self) -> Self::AijIterMut;
}

#[cfg(feature = "mpi")]
/// Traits for distributed sparse matrices.
pub trait DistributedSparseMat<'a>: BaseItem + Shape<2> {
    /// The type of the local space matrix.
    type LocalSparseMat: SparseMat<Item = Self::Item>;

    /// The type of the communicator.
    type Comm: mpi::traits::Communicator;

    /// Return the index layout.
    fn index_layout(&self) -> &IndexLayout<'a, Self::Comm>;

    /// Return the local sparse matrix.
    fn local(&self) -> &Self::LocalSparseMat;
}
