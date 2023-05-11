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
//! Matrix-matrix products are implemented through the [matrixmultiply](matrixmultiply)
//! crate. We are in the process of implementing more advanced linear algebra routines
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

pub mod data_container;
pub mod examples;
pub mod layouts;
pub mod macros;
pub mod matrix;
pub mod traits;
pub mod types;

pub mod addition;
pub mod base_matrix;
pub mod global;
pub mod matrix_multiply;
pub mod matrix_ref;
pub mod scalar_mult;
pub mod subtraction;

pub use global::*;
