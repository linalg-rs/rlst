//! Distributed computations in RLST
//!
//! RLST supports MPI distributed arrays and sparse matrices. Moreover,
//! it provides a number of utility routines for working with distributed data layouts.
//!
//! # Index Layouts
//!
//! The basic structure to store information about data distributions is the [IndexLayout](crate::distributed_tools::IndexLayout).
//! It stores an MPI communicator `comm` and a vector `scan` of length `1 + nprocs` where `nprocs` is the number of processes in the MPI
//! communicator. The range `scan[i]..scan[i + 1]` contains the indices on rank `i` (excluding the index `scan[i + 1]`).
//!
//! The easiest way to create an index layout is to use the function [IndexLayout::from_local_counts](crate::distributed_tools::IndexLayout::from_local_counts).
//! Each process calls this function with the number of local variables. They then communicate their local count to all other processes so that each process
//! can construct the complete `scan` vector.
//!
//! It is also possible to construct an index layout from [IndexLayout::from_equidistributed_chunks](crate::distributed_tools::IndexLayout::from_equidistributed_chunks). This expects the `nchunks` describing the overall number of chunks across all processes and a parameter `chunk_size` that specifies how many indices
//! are associated with each chunk. This is useful for example when we want to distribute n vectors with 3 elements each. Then `nchunks` is `n` and `chunk_size`
//! is 3. The resulting distribution will ensure that each process has a multiple of 3 local indices.
//!
//!
//!
//! # Distributed arrays
//!
//! RLST provides a [DistributedArray](crate::sparse::distributed_array::DistributedArray) type.
//! This consists of an [IndexLayout](crate::distributed_tools::IndexLayout) and a local [Array](crate::dense::array::Array) type.
//! Data is always distributed along the first dimension, meaning that for e.g. distributed matrices each process owns a number of consecutive rows of the matrix. Within a distributed array the `IndexLayout` is always stored as `RefCell`. The reason is that one frequently wishes to have mutable
//! access to the data of the distributed array while at the same time having non-mutable access to the index layout. The standard access model of Rust does not allow to have a mutable reference to one member of a struct while having non-mutable references to other members. With a `RefCell` this can be circumvented by simply cloning the `RefCell` of the index layout.
//!
//! Distributed arrays support most operations from standard arrays whereas the result always refers to the global array, e.g. the method `inner` performs an inner product across all processes, or `shape` returns the global shape of the distributed array.
//!
//! For distributed arrays the following methods are available to distribute data to multiple processes or gather data on a single process.
//!
//! - [DistributedArray::gather_to_all](crate::sparse::distributed_array::DistributedArray::gather_to_all):
//!   Gather the global array into local arrays on each process.
//! - [DistributedArray::gather_to_one](crate::sparse::distributed_array::DistributedArray::gather_to_one):
//!   Gather a global array to a single process `root`. Call this on non-root ranks.
//! - [DistributedArray::gather_to_one_root](crate::sparse::distributed_array::DistributedArray::gather_to_one):
//!   Gather a global array to a single process `root`. Call this on the root rank.
//! - [Array::scatter_from_one](crate::dense::array::Array::scatter_from_one):
//!   Create a distributed array by scattering out an array from a single process onto all processes,
//!   distributing the first dimension as determined by `IndexLayout`. Call from the non-root ranks.
//! - [Array::scatter_from_one_root](crate::dense::array::Array::scatter_from_one_root):
//!   Create a distributed array by scattering out an array from a single process onto all processes,
//!   distributing the first dimension as determined by `IndexLayout`. Call from the root rank.
//!
//! # Distributed sparse matrices
//!
//! RLST supports distributed CSR matrices via the [DistributedCsrMatrix](crate::sparse::distributed_csr_mat::DistributedCsrMatrix)
//! that operate on distributed arrays. A distributed matrix is defined by a domain layout and a range layout.
//! The domain layout determines the distribution of vectors that are multiplied with the sparse matrix and the range layout determines the distribution of
//! the rows of the distributed sparse matrix. To create a distributed sparse matrix the method
//! [DistributedCsrMatrix::from_aij](crate::traits::sparse::FromAijDistributed::from_aij)
//! from the [FromAijDistributed](crate::traits::sparse::FromAijDistributed) trait. Each process can contribute arbitrary global elements, also for rows that
//! are not associated with the current rank. The constructor routines ensures that elements are communicated to the correct process. Multiple elements associated
//! with the same global row and column are summed up on creation.
//!
//! # Distributed communication patterns
//!
//! RLST has support routines for a number of distributed communication patterns.
//!
//! ## Tools for distributed arrays
//!
//! - [sort_to_bins](crate::distributed_tools::array_tools::sort_to_bins): Take a sorted sequence and distribute it into half-open bins. This is useful
//!   for redistributing data across MPI ranks when the data distribution changes.
//! - [all_to_allv](crate::distributed_tools::array_tools::all_to_allv): A simplified MPI all_to_all communication routine.
//! - [scatterv](crate::distributed_tools::array_tools::scatterv): A simplified MPI scatter communication routine.
//! - [scatterv_root](crate::distributed_tools::array_tools::scatterv_root): A simplified MPI scatter communication routine. Call from root.
//!
//! ## Ghost communicators
//!
//! We are often in situations where we need data associated with indices on other processes. These indices are called ghost indices.
//! Underlying this is the pattern of a communication graph in which a process `j` receives data from all processes that are original owners of
//! ghost indices that `j` requires and in which `j` sends out data to all processes who have ghost indices from process `j`. A ghost communicator
//! is instantiated by calling [GhostCommunicator::new](crate::distributed_tools::ghost_communicator::GhostCommunicator::new) with the global indices
//! of the ghosts that the process needs and the owners of those indices. Once instantiated ghost data can be communicated with the following methods.
//!
//! - [GhostCommunicator::forward_send_values](crate::distributed_tools::ghost_communicator::GhostCommunicator::forward_send_values):
//!   Each process forward sends data to the processes that require them.
//! - [GhostCommunicator::forward_send_values_by_chunks](crate::distributed_tools::ghost_communicator::GhostCommunicator::forward_send_values_by_chunks):
//!   Forward send values by chunks. Use this if more than one data item is associated with an index.
//! - [GhostCommunicator::backward_send_values](crate::distributed_tools::ghost_communicator::GhostCommunicator::backward_send_values):
//!   Send data from the ghost processes backwards to the owners.
//! - [GhostCommunicator::backward_send_values_by_chunks](crate::distributed_tools::ghost_communicator::GhostCommunicator::backward_send_values_by_chunks):
//!   Send data from the ghost processes backwards to the owners by chunks. Use this if more than one data item is associated with an index.
//!
//! A complete worked out example for the use of the ghost communicator can be found in `examples/ghost_communicator.rs`.
//!
//! ## Mapping global to local data
//!
//! Imagine a global index set with indices from 0 to 9.  The first process may need the indices `0, 1, 2, 3, 4, 5, 6`, and the second process
//! may need the indices `5, 6, 7, 8, 9`. Notice that 5 and 6 are repeated for both processes.
//! The [Global2LocalDataMapper](crate::distributed_tools::data_mapper::Global2LocalDataMapper) manages this communication
//! pattern. It is instantiated with [Global2LocalDataMapper::new](crate::distributed_tools::data_mapper::Global2LocalDataMapper) which takes the
//! index layout and the required global dofs.
//! The method [Global2LocalDataMapper::map_data](crate::distributed_tools::data_mapper::Global2LocalDataMapper::map_data) then maps a vector of global data
//! to the required local data. A chunk size can be given if each index is associated with multiple elements in the data vector.
//!
//! ## Data permutations
//!
//! Consider an index layout in which the first process owns indices `0, 1, 2, 3, 4` and the second process owns the indices `5, 6, 7, 8, 9`.
//! We want to create a new index layout in which the first process owns `0, 2, 4, 6, 8` and the second process owns `1, 3, 5, 7, 9`. Notice that this is
//! a permutation of the original layout. Each index still occurs exactly once. The [DataPermutation](crate::distributed_tools::permutation::DataPermutation)
//! type provides [forward_permute](crate::distributed_tools::permutation::DataPermutation::forward_permute) and a
//! [backward_permute](crate::distributed_tools::permutation::DataPermutation::backward_permute) method to map data from a given index layout to a custom
//! permutation and back.
