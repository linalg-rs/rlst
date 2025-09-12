//! Matrix decompositions and advanced linear algebra operations
//!
//! RLST uses Lapack to provide matrix decompositions and some other advanced linear algebra
//! operations.
//!
//! # LU Decomposition and linear systems of equations
//!
//! The LU decomposition computes a decomposition of an `m x n` matrix `A` of the form
//! `A = P * L * U`. Here, `P` is an `m x m` permutation matrix, `L` is an `m x k` lower
//! triangular matrix with unit diagonal. `U` is upper triangular with dimension `k x n`.
//! Here, `k = min(m, n)`. The following example computes an LU decomposition. It requires
//! the trait [Lu](crate::Lu) to be imported.
//!
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::Lu;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! a.fill_from_standard_normal(&mut rng);
//! let lu = a.lu().unwrap();
//! ```
//! We can get the factors of the LU decomposition as follows.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::Lu;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! # a.fill_from_standard_normal(&mut rng);
//! # let lu = a.lu().unwrap();
//! let p = lu.p_mat();
//! let l = lu.l_mat();
//! let r = lu.u_mat();
//! ```
