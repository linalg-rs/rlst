//! A Rust native linear algebra library
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

// pub mod base_array;
// pub mod base_matrix;
// pub mod data_container;
// pub mod examples;
// pub mod global;
pub mod array;
pub mod layout;
pub mod macros;
//pub mod traits;
// pub mod matrix;
pub mod matrix_multiply;
// pub mod matrix_ref;
// pub mod matrix_view;
// pub mod op_containers;
// pub mod traits;
// pub mod types;

// pub use global::*;
