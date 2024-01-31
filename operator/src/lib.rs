//! The operator library contains traits and methods to support arbitrary operator types.
#![cfg_attr(feature = "strict", deny(warnings))]

pub mod implementation;
pub mod linalg;
pub mod operations;
pub mod operator;
pub mod space;

pub use operator::*;
pub use space::*;
