//! Definition of Index Layouts.
//!
//! An [IndexLayout] specified how degrees of freedom are distributed among processes.
//! We always assume that a process has a contiguous set of degrees of freedom.

use rlst_dense::types::RlstResult;

/// The Index Layout trait. It fully specifies how degrees of freedom are distributed
/// among processes. Each process must hold a contiguous number of degrees of freedom (dofs).
/// However, it is possible that a process holds no dof at all. Local indices are specified by
/// index ranges of the type [first, last). The index `first` is contained on the process. The
/// index `last` is not contained on the process. If `first == last` then there is no index on
/// the local process.
pub trait IndexLayout {
    /// The local index range. If there is no local index
    /// the left and right bound are identical.
    fn local_range(&self) -> (usize, usize);

    /// The number of global indices.
    fn number_of_global_indices(&self) -> usize;

    /// The number of local indicies, that is the amount of indicies
    /// on my process.
    fn number_of_local_indices(&self) -> usize;

    /// Index range on a given process.
    fn index_range(&self, rank: usize) -> RlstResult<(usize, usize)>;

    /// Convert continuous (0, n) indices to actual indices.
    ///
    /// Assume that the local range is (30, 40). Then this method
    /// will map (0,10) -> (30, 40).
    /// It returns ```None``` if ```index``` is out of bounds.
    fn local2global(&self, index: usize) -> Option<usize>;

    /// Convert global index to local index on a given rank.
    /// Returns ```None``` if index does not exist on rank.
    fn global2local(&self, rank: usize, index: usize) -> Option<usize>;

    /// Get the rank of a given index.
    fn rank_from_index(&self, index: usize) -> Option<usize>;
}
