//! The operator library contains traits and methods to support arbitrary operator types.
#![cfg_attr(feature = "strict", deny(warnings))]
#![warn(missing_docs)]

pub mod abstract_operator;
pub mod interface;
pub mod linalg;
pub mod operations;
pub mod space;

pub use abstract_operator::*;
pub use space::*;
