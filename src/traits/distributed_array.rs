//! Traits for distributed arrays.

use std::rc::Rc;

use mpi::traits::Communicator;

use crate::distributed_tools::IndexLayout;

/// Gather array to all processes.
pub trait GatherToAll {
    /// The output type after gathering the distributed array.
    type Output;
    /// Gather the distributed array to all processes.
    fn gather_to_all(&self) -> Self::Output;
}

/// Gather array to one process.
pub trait GatherToOne {
    /// The output type after gathering the distributed array to one process.
    type Output;
    /// Gather the distributed array to one process.
    fn gather_to_one(&self, root: usize);
    /// Call this on the root process that will receive the data.
    fn gather_to_one_root(&self) -> Self::Output;
}

/// Scatter array from one process to all processes.
///
/// This trait should be implemented on a standard array and produce a distributed array.
pub trait ScatterFromOne {
    /// The output type after scattering the distributed array.
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
