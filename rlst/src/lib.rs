//! Interface to the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod dense;

pub mod prelude;
pub mod threading;

pub use prelude::*;
