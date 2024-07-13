//! Initialising RLST.
//!
//! To use RLST add it as usual dependency to your `Cargo.toml`. To simply import all RLST symbols into your namespace
//! use the command.
//! ```
//! use rlst::prelude::*;
//! ```
//! By default RLST uses the system provided BLAS/Lapack libraries. On Linux this is `libblas` and `liblapack`. On Mac
//! it is the `Accelerate` framework. It is possible to use custom BLAS libraries. But this feature is not yet stable.
