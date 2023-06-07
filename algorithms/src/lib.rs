//! Collection of Linear Solver Algorithms and Interfaces
#![cfg_attr(feature = "strict", deny(warnings))]

extern crate rlst_blis_src;

pub mod dense;
pub mod iterative_solvers;
pub mod lapack;
pub mod linalg;
pub mod traits;
