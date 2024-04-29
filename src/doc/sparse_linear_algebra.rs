//! Sparse linear algebra with RLST.
//!
//! RLST has basic support for sparse linear algebra. Currently, we are supporting
//! CSR and CSC matrices, matrix-vector products with these matrices and LU decomposition via
//! [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html).
//!
//! # Creating sparse matrices
//!
//! A sparse matrix in CSR or CSC format can easily be created as follows.
//! ```
//! # use rlst::prelude::*;
//! let rows = vec![0, 0, 1, 1];
//! let cols = vec![0, 1, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//!
//! let csc = CscMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();
//! let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();
//! ```
//! The [CscMatrix](crate::CscMatrix) and [CsrMatrix](crate::CsrMatrix) take the
//! shape, row indices, column indices, and associated data.
//!
//! If the data is already available in CSR or CSC format one can also directly use
//! the `new` method of these types to create the corresponding matrices without the need
//! for data conversion.
//!
//! # Sparse matrix-vector products
//!
//! To multiply a matrix with a vector use the following code.
//! ```
//! // Test the matrix [[1, 2], [3, 4]]
//! use rlst::prelude::*;
//! let rows = vec![0, 0, 1, 1];
//! let cols = vec![0, 1, 0, 1];
//! let data = vec![1.0, 2.0, 3.0, 4.0];
//!
//! let csr = CsrMatrix::from_aij([2, 2], &rows, &cols, &data).unwrap();
//!
//! // Execute 2 * [1, 2] + 3 * A*x with x = [3, 4];
//! // Expected result is [35, 79].
//!
//! let x = vec![3.0, 4.0];
//! let mut res = vec![1.0, 2.0];
//!
//! csr.matmul(3.0, &x, 2.0, &mut res);

//! assert_eq!(res[0], 35.0);
//! assert_eq!(res[1], 79.0);
//! ```
//!
//! # LU decomposition
//!
//! We supoprt LU decomposition via the external [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
//! library from the `Suitesparse` package. To enable `Umfpack` use the feature flag `suitesparse`. Please note
//! that currently `Umfpack` is GPL license, which may impact the license of binaries linked with
//! it. Note that `Umfpack` requires linkage with Blas and Lapack. Hence, code needs to define
//! somewhere the statements `extern crate blas_src` and `extern crate lapack_src` to make sure
//! that Blas and Lapack are linked during compilation.
//! ```
//! # extern crate blas_src;
//! # extern crate lapack_src;  
//! # use rlst::prelude::*;
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
//! The `into_lu` method is provided for CSC and CSR matrices. However, since Umfpack only supports
//! CSC matrices CSR data is internally converted to CSC data when calling `into_lu`. This incurs
//! some overhead in terms of performance and memory consumption.
