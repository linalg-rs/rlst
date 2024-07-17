//! The RLST Operator Interface.
//!
//! RLST supports linear algebra on abstract function spaces. The idea is
//! that we define a linear space and operators acting on this space. Algorithms
//! operating on this abstract interface will then be automatically usuable for
//! any linear structure that supports this interface.
//!
//! # Function Spaces
//!
//! The basis of the abstract operator interface is the [LinearSpace](crate::LinearSpace).
//! A linear space is essentially factory that provides
