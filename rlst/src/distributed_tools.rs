//! Useful tools for arrays on MPI systems.

pub mod array_tools;
pub mod data_mapper;
pub mod ghost_communicator;
pub mod index_embedding;
pub mod index_layout;
pub mod permutation;

pub use array_tools::{
    all_to_all_varcount, all_to_allv, displacements, scatterv, scatterv_root, sort_to_bins,
};
pub use data_mapper::Global2LocalDataMapper;
pub use ghost_communicator::GhostCommunicator;
pub use index_layout::IndexLayout;
pub use permutation::DataPermutation;
