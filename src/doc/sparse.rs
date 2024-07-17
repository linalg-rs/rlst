//! Sparse matrices in RLST.
//!
//! RLST has preliminary support for CSR and CSC sparse matrices.
//! To initialise a CSR sparse matrix use the following code.
//! ```
//! use rlst::prelude::*;
//! use rand::Rng;
//! let rows = vec![0, 0, 1, 1];
//! let cols = vec![0, 1, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//!
//! let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();
//! let x = vec![3.0, 4.0];
//! let mut res = vec![1.0, 2.0];
//!
//! csr.matmul(3.0, &x, 2.0, &mut res);
//!
//! assert_eq!(res[0], 35.0);
//! assert_eq!(res[1], 79.0);
//! ```
//! To use the `csc` format instead of the `csr` format use the [CscMatrix](crate::CscMatrix) structure.
//!
//! In addition to simple matrix-vector products RLST supports solution of sparse linear systems using the
//! Umfpack solver contained within [Suitesparse](https://people.engr.tamu.edu/davis/suitesparse.html).
//! The following code demonstrates how to use this solver.
//!
//! ```
//! let n = 5;
//!
//! let mut mat = rlst_dynamic_array2!(f64, [n, n]);
//! let mut x_exact = rlst_dynamic_array1!(f64, [n]);
//! let mut x_actual = rlst_dynamic_array1!(f64, [n]);
//!
//! mat.fill_from_seed_equally_distributed(0);
//! x_exact.fill_from_seed_equally_distributed(1);
//!
//! let rhs = empty_array::<f64, 1>().simple_mult_into_resize(mat.view(), x_exact.view());
//!
//! let mut rows = Vec::<usize>::with_capacity(n * n);
//! let mut cols = Vec::<usize>::with_capacity(n * n);
//! let mut data = Vec::<f64>::with_capacity(n * n);
//!
//! for col_index in 0..n {
//!     for row_index in 0..n {
//!         rows.push(row_index);
//!         cols.push(col_index);
//!         data.push(mat[[row_index, col_index]]);
//!     }
//! }
//!
//! let sparse_mat = CscMatrix::from_aij([n, n], &rows, &cols, &data).unwrap();
//!
//! sparse_mat
//!     .into_lu()
//!     .unwrap()
//!     .solve(rhs.view(), x_actual.view_mut(), TransMode::NoTrans)
//!     .unwrap();
//!
//! rlst::assert_array_relative_eq!(x_actual, x_exact, 1E-12);
//! ```
//! Sparse matrices do not yet support algebraic operations such as multiplications with scalars, additions, etc.
//! To use those it is possible to wrap a sparse matrix into an abstract operator and use the abstract operator interface.
