//! Dense array types and operations on them.
//!
#![cfg_attr(feature = "strict", deny(warnings))]

extern crate rlst_blis_src;
extern crate rlst_netlib_lapack_src;

pub mod base_array;
pub mod data_container;
pub mod linalg;
pub mod number_types;
pub mod tools;
pub mod traits;

pub mod array;
pub mod gemm;
pub mod layout;
pub mod macros;
pub mod matrix_multiply;
pub mod types;
