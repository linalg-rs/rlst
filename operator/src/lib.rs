//! The operator library contains traits and methods to support arbitrary operator types.
#![cfg_attr(feature = "strict", deny(warnings))]

pub mod linalg;
pub mod operator;
pub mod spaces;

pub use operator::*;
pub use spaces::*;
