//! Common Examples
//!
//! # Defining matrices and vectors
//!
//! We can define a simple 3 x 5 matrix with `f64` entries
//! as follows.
//!
//! ```
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5)];
//! ```
//! The result is a row-major matrix. To define a column-major
//! matrix we use
//!
//! ```
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5), ColumnMajor];
//! ```
//! A new column vector can be created by
//! ```
//! # use rlst_dense::*;
//! let mut vec = vector![f64, 5];
//! ```
//!
//! For a row vector use
//! ```
//! # use rlst_dense::*;
//! let mut vec = vector![f64, 5, RowVector];
//! ```
//!
//! A normally distributed random matrix is obtained as
//! ```
//! # use rlst_dense::*;
//! let mat = rand_mat![f64, (3, 5)];
//! ```
//! This is identical to the following commands.
//! ```
//! # use rlst_dense::*;
//! let mut rng = rand::thread_rng();
//! let mut mat = mat![f64, (3, 5)];
//! mat.fill_from_rand_standard_normal(&mut rng);
//! ```
//! # Accessing entries
//!
//! We can access and modify entries of a matrix
//! with bounds checks in the following way.
//! ```
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5)];
//! *mat.get_mut(2, 3) = 4.0;
//! assert_eq!(mat.get(2, 3), 4.0);
//! ```
//! The methods `get` and `get_mut` perform an assertion
//! check if the provided row and column are inside bounds.
//! If not the assertion fails. The following two codes will produce
//! an exception.
//! ```should_panic
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5)];
//! *mat.get_mut(3, 5) = 4.0;
//! ```
//!
//! ```should_panic
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5)];
//! println!("Print out of bounds entry: {}",mat.get(3, 5));
//! ```
//! Bounds checks are not always desired. We therefore also provide
//! unsafe access routines.
//! ```
//! # use rlst_dense::*;
//! let mut mat = mat![f64, (3, 5)];
//! unsafe {
//!     *mat.get_unchecked_mut(2, 3) = 4.0;
//!     assert_eq!(mat.get_unchecked(2, 3), 4.0);
//! }
//! ```
//! # Operations on matrices and vectors.
//!
//! Matrix/vector sums or products with a scalar are written as
//! ```
//! # use rlst_dense::*;
//! let mat1 = rand_mat![f64, (3, 5)];
//! let mat2 = rand_mat![f64, (3, 5)];
//! let sum = (3.0 * &mat1 + &mat2).eval();
//! assert_eq!(sum.get(2, 4), 3.0 * mat1.get(2, 4) + mat2.get(2, 4));
//! ```
//! Note the `eval` statement at the end. If we were
//! only to write `3.0 * &mat1 + &mat2` the result would
//! be a type that stores references to `mat1`, `mat2`, and
//! the associated operations. But it would not execute those.
//! The `eval` statement instantiates a new matrix and then iterates
//! through the new matrix to componentwise fill up the matrix with
//! the result of the right-hand side componentwise operation.
//!
//! Matrix/vector and matrix/matrix products are implemented via
//! the [matrixmultiply] crate.
//! ```
//! # use rlst_dense::*;
//! let mat = rand_mat![f64, (3, 5)];
//! let col_vec = rand_vector![f64, 5];
//! let row_vec = rand_vector![f64, 3, RowVector];
//! let res1 = mat.dot(&col_vec);
//! let res2 = row_vec.dot(&mat);
//! ```
//! # Access to submatrices.
//!
//! We can access a single subblock of a matrix as follows.
//! ```
//! # use rlst_dense::*;
//! let mat = rand_mat![f64, (10, 10)];
//! let block = mat.block((2, 2), (3, 5));
//! assert_eq!(block.get(2, 4), mat.get(4, 6))
//! ```
//! The variable `block` is now a read-only view onto the submatrix
//! of `mat` starting at position `(2, 2)` with dimensions `(3, 5)`.
//! If we want to have a mutable subblock the corresponding method
//! is `mat.block_mut`. However, in this way we can only create one mutable
//! view onto the matrix. Quite often we want to split a matrix into multiple
//! mutable blocks that can be independently accessed. This can be achieved
//! with the method `mat.split_in_four_mut` as the following example demonstrates.
//! ```
//! # use rlst_dense::*;
//! let mut mat = rand_mat![f64, (10, 10)];
//! let (mut m1, mut m2, mut m3, mut m4) = mat.split_in_four_mut((5, 5));
//! *m1.get_mut(1, 0) = 2.0;
//! *m2.get_mut(3, 4) = 3.0;
//! *m3.get_mut(2, 1) = 4.0;
//! *m4.get_mut(4, 2) = 5.0;
//! assert_eq!(mat.get(1, 0), 2.0);
//! assert_eq!(mat.get(3, 9), 3.0);
//! assert_eq!(mat.get(7, 1), 4.0);
//! assert_eq!(mat.get(9, 7), 5.0);
//! ```
//! The matrix is split up into 4 submatrices with the first one starting at
//! (0, 0) and the last one starting at (5, 5). The second block is correspondingly
//! the upper right submatrix and the third block is the lower left submatrix.
//! All 4 matrices are independently accessible.
