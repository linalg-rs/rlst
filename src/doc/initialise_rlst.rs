//! Initialising RLST.
//!
//! To use RLST add it as usual dependency to your `Cargo.toml`. To simply import all RLST symbols into your namespace
//! use the command.
//! ```
//! use rlst::prelude::*;
//! ```
//! Several RLST functions depend on the availability of a BLAS or Lapack library.
//! We recommend to use the packages [blas-src] and [lapack-src]. For example on a Mac system
//! add the following dependencies to your project.
//! ```text
//! blas-src = { version = "0.9", features = ["accelerate"]}
//! lapack-src = { version = "0.9", features = ["accelerate"]}
//! ````
//! This uses Apple Accelerate as BLAS/Lapack provider. On a Linux system you may want to use [BLIS](https://github.com/flame/blis)
//! or [OpenBLAS](https://www.openblas.net) instead. To use BLIS you would add the following lines.
//! ```text
//! blas-src = { version = "0.9", features = ["blis"]}
//! lapack-src = { version = "0.9", features = ["netlib"]}
//! ````
//! For OpenBLAS you would choose
//! ```text
//! blas-src = { version = "0.9", features = ["openblas"]}
//! lapack-src = { version = "0.9", features = ["openblas"]}
//! ````
//! You may only want to include BLAS into your `dependencies` for a final application. If you are developing a library that uses
//! RLST it may be useful to include the Blas/Lapack providers only as `dev-dependencies` so that users of your library can
//! then decide which provider to use.
//!
//! You will also need to include somewhere in your library a reference to the Blas/Lapack source packages. Otherwise,
//! the corresponding symbols may not be linked to your compilation, leading to missing symbol errors when compiling. To do this add the
//! following two lines, e.g. in your `lib.rs` or `main.rs` file.
//! ```
//! extern crate blas_src;
//! extern crate lapack_src;
//! ````
