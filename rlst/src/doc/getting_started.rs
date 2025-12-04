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
//!
//! ## Using the FFT interface
//!
//! RLST optionally links with FFTW to provide FFT for one, two, and three dimensional arrays.
//! Note that **FFTW is GPL licensed.**. This means that any binary linked against FFTW must be
//! distributed under the GPL license as well. To enable the FFT use the feature flag `fftw`.
//! The actual `fftw` linkage can be done through one of the following options:
//! - Use the `fftw_system` feature flag to link against a system installation of FFTW.
//!   The library must be in a standard search path of the Rust compiler.
//! - Use the `fftw_source` flag to build FFTW and statically link it.
//! - Use the `fftw_mkl` flag to link against Intel MKL, which provides an FFTW compatible interface.
//!   For this option the license conditions of Intel MKL rather than FFTW apply.
//!
//! Instructions on how to use the FFT can be found in the documentation of `rlst::dense::fftw` when
//! the `fftw` feature flag is activated.
