//! A Rust native linear algebra library
//!
//! The goal of `householder` is the development of a Rust native
//! linear algebra library that is performant and does not require
//! external BLAS/Lapack linkage or dependencies.
//!
//! The library is early stage. The core data structures are implemented
//! and functionality is continuously being added.
//!
//! The core of `householder` is the [Matrix](crate::matrix::Matrix) type,
//! which supports fixed size implementations, dynamic size implementations,
//! and specialises also to vectors. It is agnostic to underlying storage
//! layouts (i.e. row or column major) and also supports layouts with
//! arbitrary strides.
//!
//! Algebraic operations on matrices are implemented using a system that is akin
//! to expression templates. Adding matrices or multiplying them with a scalar
//! is not immediately executed but creates a new type that stores the information
//! about the operation. Only when the user asks for the evaluation, all operations
//! are executed in a single pass without creating temporaries.
//!
//! Matrix-matrix products are implemented through interfaces to Blis.
//! We are in the process of implementing more advanced linear algebra routines
//! (e.g. LU, QR, etc.). But these are not yet available. The focus is on implementing
//! modern blocked multi-threaded routines whose performance is competitive with Lapack.
//!
//! To learn more about `householder` we recommend the user to read the following bits
//! of information.
//!
//! - [Basic trait definitions](crate::traits)
//! - [Matrix storage layouts](crate::layouts)
//! - [The Matrix type](crate::matrix)
//! - [Examples](crate::examples)
#![cfg_attr(feature = "strict", deny(warnings))]

extern crate rlst_blis_src;
extern crate rlst_netlib_lapack_src;

pub mod base_array;
pub mod data_container;
pub mod linalg;
pub mod number_types;

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
