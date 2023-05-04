//! Collection of Linear Solver Algorithms and Interfaces
#![cfg_attr(feature = "strict", deny(warnings))]

#[allow(unused_imports)]
use lapack_src;

pub mod iterative_solvers;
pub mod lapack;
pub mod traits;
