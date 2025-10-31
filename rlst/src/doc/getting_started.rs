//! Getting started.
//!
//! ## Selecting a BLAS library
//!
//! RLST depends on a correctly configured BLAS/Lapack environment. On Mac OS the following should
//! be added to `Cargo.toml` for a project that uses RLST.
//! ```json
//! blas-src = { version = "0.3", features = ["accelerate"]}
//! lapack-src = { version = "0.11", features = ["accelerate"]}
//! ```
//! This links the library with Apple's Accelerate. On Linux it depends
//! what the environment provides. Many distributions by default use OpenBLAS,
//! which can be linked with
//! ```json
//! blas-src = { version = "0.3", features = ["openblas"]}
//! ```
//! OpenBlas is frequently already linked with Lapack. If not, a suitable `lapack-src`
//! call similar to Mac OS is also required.
//!
//! Within the code that uses RLST then place within your  source files the lines
//! ```no_run
//! extern crate blas_src;
//! ```
//! and if necessary also
//! ```no_run
//! extern crate lapack_src;
//! ```
//! This ensures that the compiler links your code with the provided Blas/Lapack libraries.
//!
