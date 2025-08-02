//! Traits for distributed patterns.
//!
//! The traits in this module provide typical patterns for working with distributed array data.

use std::rc::Rc;

use mpi::traits::Communicator;

use crate::distributed_tools::IndexLayout;

/// Gather to all processes.
pub trait GatherToAll {
    /// The output type after gathering the distributed data.
    type Output;
    /// Gather the distributed data to all processes.
    fn gather_to_all(&self) -> Self::Output;
}

/// Gather to one process.
pub trait GatherToOne {
    /// The output type after gathering the distributed array to one process.
    type Output;
    /// Gather the distributed array rank `root`.
    ///
    /// Call this method on all ranks that are not `root`.
    fn gather_to_one(&self, root: usize);
    /// Gather the distributed array rank `root`.
    ///
    /// Call this on the root rank that will receive the data.
    fn gather_to_one_root(&self) -> Self::Output;
}

/// Scatter data from one process to all processes.
pub trait ScatterFromOne {
    /// The output type after scattering the data.
    type Output<'a, C>
    where
        C: 'a,
        C: Communicator;
    /// Scatter the distributed array from one process to all processes.
    ///
    /// Call this on the root process that will send the data.
    fn scatter_from_one_root<'a, C: Communicator>(
        &self,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C>;

    /// Scatter the distributed array from one process to all processes.
    ///
    /// Call this on all receivers of data.
    fn scatter_from_one<'a, C: Communicator>(
        root: usize,
        index_layout: Rc<IndexLayout<'a, C>>,
    ) -> Self::Output<'a, C>;
}
