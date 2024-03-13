//! Interface to the rlst library

extern crate blas_src;
extern crate lapack_src;

pub use rlst_dense as dense;

pub use rlst_sparse as sparse;

pub use rlst_operator as operator;

pub mod prelude;
pub mod threading;
