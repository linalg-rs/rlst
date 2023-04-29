//! Trait definitions
//!
//! `householder` relies heavily on traits. This module
//! collects the core trait definitions to define a matrix type.
//! The following functionality is available through traits.
//!
//! - [Definition of random access operations.](random_access)
//! - [Size type descriptions (e.g. fixed given dimenion or dynamically allocated).](size)
//! - [Matrix storage layout.](layout)
//! - [Summary trait defining a matrix.](matrix)

pub mod layout;
pub mod matrix;
pub mod size;

pub use layout::*;
pub use matrix::*;
pub use rlst_common::traits::accessors::*;
pub use rlst_common::traits::properties::*;
pub use size::*;
