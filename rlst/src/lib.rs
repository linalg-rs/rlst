//! Interface to the rlst library
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub use rlst_dense as dense;

pub use rlst_sparse as sparse;

pub use rlst_operator as operator;

pub mod prelude;
pub mod threading;

pub use prelude::*;
