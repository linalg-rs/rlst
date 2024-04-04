//! Interface to the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod dense;
pub mod external;
pub mod io;
pub mod sparse;

pub mod prelude;
pub mod threading;

pub mod operator;

pub use prelude::*;

#[cfg(test)]
mod test {

    extern crate blas_src;
    extern crate lapack_src;
}
