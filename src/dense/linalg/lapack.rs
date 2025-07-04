//! Lapack interface for linear algebra operations.

use crate::base_types::{LapackError, LapackResult};

pub mod cholesky;
pub mod eigenvalue_decomposition;
pub mod inverse;
pub mod lu;
pub mod pseudo_inverse;
pub mod qr;
pub mod singular_value_decomposition;
pub mod solve;
pub mod solve_triangular;
pub mod symmeig;

pub mod interface;
