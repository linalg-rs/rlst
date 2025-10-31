//! Matrix decompositions and advanced linear algebra operations
//!
//! RLST uses Lapack to provide matrix decompositions and some other advanced linear algebra
//! operations.
//!
//! - [LU Decomposition and linear systems of equations](#lu-decomposition-and-linear-systems-of-equations)
//! - [Computing the determinant](#computing-the-determinant)
//! - [Matrix inverses](#matrix-inverses)
//! - [Solvers for linear systems and least-squares problems](#solvers-for-linear-systems-and-least-squares-problems)
//! - [Triangular linear systems](#triangular-linear-systems)
//! - [Cholesky decomposition and linear system solves](#cholesky-decomposition-and-linear-system-solves)
//! - [QR Decomposition](#qr-decomposition)
//! - [Singular Value Decomposition](#singular-value-decomposition)
//! - [Pseudo-Inverse of a matrix](#pseudo-inverse-of-a-matrix)
//! - [Symmetric Eigenvalue Decomposition](#symmetric-eigenvalue-decomposition)
//! - [General Eigenvalue Decomposition](#general-eigenvalue-decomposition)
//! - [Schur decomposition](#schur-decomposition)
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
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::Lu;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! # a.fill_from_standard_normal(&mut rng);
//! # let lu = a.lu().unwrap();
//! let p = lu.p_mat().unwrap();
//! let l = lu.l_mat().unwrap();
//! let u = lu.u_mat().unwrap();
//! rlst::assert_array_relative_eq!(dot!(p, l, u), a, 1E-10);
//! ```
//! To solve a linear system of equations we use the `solve` method.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::Lu;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let lu = a.lu().unwrap();
//! let mut b = rlst::rlst_dynamic_array!(f64, [5]);
//! b.fill_from_standard_normal(&mut rng);
//! let x = lu.solve(TransMode::NoTrans, &b).unwrap();
//! ```
//! # Computing the Determinant
//!
//! We can compute the determinant of a matrix `a` from its LU decomposition.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::Lu;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let det = a.lu().unwrap().det();
//! ````
//! # Matrix inverses
//!
//! To compute the inverse of a `n x n` matrix `A` simply use the following.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::Lu;
//! use rlst::Inverse;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let inv = a.inverse().unwrap();
//! ```
//!
//! # Solvers for linear systems and least-squares problems
//!
//! We can directly solve linear systems and least-squares problems with the [solve](crate::Solve) trait
//! implemented for dense matrices. Depending on whether the system is square or rectangular LU decomposition
//! or a least-squares solver is used. The following gives an example of a least-squares solve.
//!
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::Solve;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! a.fill_from_standard_normal(&mut rng);
//! let mut b = rlst::rlst_dynamic_array!(f64, [5]);
//! b.fill_from_standard_normal(&mut rng);
//! let x = a.solve(&b).unwrap();
//! ```
//!
//! # Triangular linear systems
//!
//! We can solve linear systems with triangular matrices as follows.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::base_types::UpLo;
//! use rlst::SolveTriangular;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let mut b = rlst::rlst_dynamic_array!(f64, [5]);
//! b.fill_from_standard_normal(&mut rng);
//! let x = a.solve_triangular(UpLo::Upper, &b).unwrap();
//! ```
//! The triangular solver uses the parameter [UpLo](crate::base_types::UpLo) to determine whether to use the upper or
//! lower triangular part of `a`.
//!
//! # Cholesky decomposition and linear system solves
//!
//! If a matrix `A` is Hermitian and positve definite the Cholesky Decomposition exists as `A = C^H * C`
//! with `C` upper triangular. The following gives an example on how to compute the Cholesky decomposition.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::EvaluateObject;
//! use rlst::Cholesky;
//! use rlst::base_types::UpLo;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let a = dot!(a.r().conj().transpose().eval(), a.r());
//! let cholesky = a.cholesky(UpLo::Upper).unwrap();
//! rlst::assert_array_relative_eq!(dot!(cholesky.r().conj().transpose().eval(), cholesky.r()), a, 1E-10);
//! ```
//! The parameter [UpLo](crate::base_types::UpLo) determines whether the Cholesky decomposition uses the upper or lower
//! triangular part of the matrix. Correspondingly, the upper or lower triangular Cholesky factor is returned.
//!
//! To solve a linear system using the Cholesky decomposition use the following.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! # use rlst::CholeskySolve;
//! # use rlst::base_types::UpLo;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! # a.fill_from_standard_normal(&mut rng);
//! # let a = dot!(a.r().conj().transpose().eval(), a.r());
//! let mut b = rlst::rlst_dynamic_array!(f64, [5]);
//! b.fill_from_standard_normal(&mut rng);
//! let x = a.cholesky_solve(UpLo::Upper, &b).unwrap();
//! rlst::assert_array_relative_eq!(dot!(a, x), b, 1E-10);
//! ```
//!
//! # QR Decomposition
//!
//! The QR Decomposition of a `m x n` matrix `A` is defined as the decomposition
//! of the form `A * P = Q * R`. Here,
//! - `P` is a permutation matrix of dimension `n x n`. If pivoting is disabled then `P`
//!   is just the identity matrix.
//! - `Q` is a `m x k` matrix (`k = min(m, n)`) with orthogonal unit length columns.
//! - `R` is a upper triangular `k x n` matrix (`k = min(m, n)`). If pivoting is enabled
//!   the diagonal elements of `R` are ordered by magnitude in descending order.
//!
//! The following gives an example of using the QR decomposition.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! use rlst::Qr;
//! use rlst::dense::linalg::lapack::qr::EnablePivoting;
//! use rlst::dense::linalg::lapack::qr::QMode;
//! let mut a= rlst::rlst_dynamic_array!(f64, [5, 3]);
//! a.fill_from_standard_normal(&mut rng);
//! let qr = a.qr(EnablePivoting::Yes).unwrap();
//! let p = qr.p_mat().unwrap();
//! let q = qr.q_mat(QMode::Compact).unwrap();
//! let r = qr.r_mat().unwrap();
//! rlst::assert_array_relative_eq!(dot!(q, r, p.r().transpose().eval()), a, 1E-10);
//! ```
//! For the matrix `Q` we have used the paramter [QMode::Compact](crate::dense::linalg::lapack::qr::QMode::Compact).
//! This returns a `Q` matrix of dimension `m x k`. One can also return a full `m x m` matrix by
//! providing the parameter [QMode::Full](crate::dense::linalg::lapack::qr::QMode::Full) instead.
//!
//! # Singular Value Decomposition
//!
//! The singluar value decomposition of a `m x n` matrix `A` is a decomposition of the form
//! `A = U * S * Vt`, where
//! - `U` is an `m x m` matrix with orthogonal columns of unit length.
//! - `S` is a diagonal `m x n` matrix with `min(m, n)` singular values `s1 >= s2 >= ... >= 0` on its diagonal.
//! - `Vt` is a `n x n` matrix with orthogonal columns of unit length.
//!
//! In the following example we compute only the singular values of a matrix.
//!
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::SingularValueDecomposition;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! a.fill_from_standard_normal(&mut rng);
//! let singular_values = a.singular_values().unwrap();
//! assert_eq!(singular_values.len(), 3);
//! ```
//! We can also compute the full SVD as follows.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::SingularValueDecomposition;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! # a.fill_from_standard_normal(&mut rng);
//! use itertools::izip;
//! use rlst::dense::linalg::lapack::singular_value_decomposition::SvdMode;
//! let (s, u, vt) = a.svd(SvdMode::Full).unwrap();
//! let mut s_mat = rlst::DynArray::<f64, 2>::from_shape([5, 3]);
//! for (d, s_val) in izip!(s_mat.diag_iter_mut(), s.iter_value()) {
//!     *d = s_val;
//! }
//! rlst::assert_array_relative_eq!(dot!(u, s_mat, vt), a, 1E-10);
//! ```
//! The parameter [SvdMode::Full](crate::dense::linalg::lapack::singular_value_decomposition::SvdMode::Full) returns the full SVD as
//! defined above. To instead compute a compact representation with `m x k` matrix `U` and `k x n` matrix `Vt` use the
//! parameter [SvdMode::Compact](crate::dense::linalg::lapack::singular_value_decomposition::SvdMode::Compact).
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::SingularValueDecomposition;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! # a.fill_from_standard_normal(&mut rng);
//! # use rlst::dense::linalg::lapack::singular_value_decomposition::SvdMode;
//! use itertools::izip;
//! let (s, u, vt) = a.svd(SvdMode::Compact).unwrap();
//! let mut s_mat = rlst::DynArray::<f64, 2>::from_shape([3, 3]);
//! for (d, s_val) in izip!(s_mat.diag_iter_mut(), s.iter_value()) {
//!     *d = s_val;
//! }
//! rlst::assert_array_relative_eq!(dot!(u, s_mat, vt), a, 1E-10);
//! ```
//!
//! # Pseudo-Inverse of a matrix
//!
//! The pseudo-inverse for an `m x n` matrix with `m >= n` is given as `P = (A^H* A )^{-1} * A^H`.
//! If `m < n` then the pseudo-inverse is `P = A^H * (A * A^H)`. The pseudo-inverse can be computed
//! with the singular value decomposition. In practice the pseudo-inverse is usually regularized in the
//! sense that only a certain number of the largest singular values are used, either specified through a
//! tolerance or through a maximum number of singular values.
//!
//! The following computes the pseudo-inverse of a given matrix using a singular-value cut-off of `1E-10`
//! and no maximum number of singular values.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::SingularValueDecomposition;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [10, 5]);
//! # a.fill_from_standard_normal(&mut rng);
//! let p_inv = a.pseudo_inverse(None, Some(1E-10)).unwrap();
//! ```
//! For numerical stability reasons the pseudo-inverse is not stored as a single matrix in `p_inv` but
//! in a disassembled structure. To apply the pseudo-inverse to a vector or matrix use the following.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! use rlst::SingularValueDecomposition;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [10, 5]);
//! # a.fill_from_standard_normal(&mut rng);
//! # let p_inv = a.pseudo_inverse(None, Some(1E-10)).unwrap();
//! let mut b = rlst::rlst_dynamic_array!(f64, [10]);
//! b.fill_from_standard_normal(&mut rng);
//! let out = p_inv.apply(&b);
//! ```
//!
//!
//! # Symmetric Eigenvalue Decomposition
//!
//! For a real symmetric or complex Hermitian `n x n` matrix `A` the eigenvalue
//! decomposition is given as `A = Q * L * Q^H` with `Q` a matrix with orthogonal
//! columns of unit length, and `Q^H` the complex conjugate transpose of `Q`. An important
//! property of this eigenvalue problem is that all eigenvalues are real. We can compute the
//! eigenvalues as follows.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::dot;
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! use rlst::dense::linalg::lapack::symmeig::SymmEigMode;
//! use rlst::SymmEig;
//! use rlst::base_types::UpLo;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let a = (a.r() + a.r().transpose()).eval();
//! let (eig, _) = a.eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly).unwrap();
//! ```
//! To compute the eigenvalues and the eigenvector matrix `Q` use the following.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::{diag, dot};
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::dense::linalg::lapack::symmeig::SymmEigMode;
//! # use rlst::SymmEig;
//! # use rlst::base_types::UpLo;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! # use rlst::EvaluateObject;
//! a.fill_from_standard_normal(&mut rng);
//! let a = (a.r() + a.r().transpose()).eval();
//! let (eig, Q) = a.eigh(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors).unwrap();
//! let Q = Q.unwrap();
//! let eig_mat = diag!(eig);
//! rlst::assert_array_relative_eq!(a, dot!(Q.r(), eig_mat, Q.r().transpose().eval()), 1E-10);
//! ```
//! # General Eigenvalue Decomposition
//!
//! The general eigenvalue decomposition for a `n x n` matrix `A` is of the form
//! `A * Vr = Vr * L` for a diagonal matrix `L` containing the eigenvalues
//! and a matrix `Vr` whose columns are the right eigenvectors. Alternatively,
//! one may be interested in the left eigenvectors given as the columns of the matrix `Vl`
//! satisfying `Vl^H * A = L * Vl^H` with `Vl^H` being the complex conjugate transpose of `A`.
//!
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::{dot, diag};
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! use rlst::base_types::c64;
//! use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
//! use rlst::EigenvalueDecomposition;
//! use rlst::Inverse;
//! let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! a.fill_from_standard_normal(&mut rng);
//! let (lam, vr, vl) = a.eig(EigMode::BothEigenvectors).unwrap();
//! let vr = vr.unwrap();
//! let vlh = vl.unwrap().conj().transpose().eval();
//! let lam_mat = diag!(lam);
//! rlst::assert_array_abs_diff_eq!(dot!(vr, lam_mat, vr.inverse().unwrap()), a.r().into_type::<c64>(), 1E-10);
//! rlst::assert_array_abs_diff_eq!(dot!(vlh.inverse().unwrap(), lam_mat, vlh), a.r().into_type::<c64>(), 1E-10);
//! ```
//! Note that for the assertion at the end we convert the matrix `a` to the complex type [c64](crate::base_types::c64).
//! The reason is that eigenvalues and eigenvectors are complex in general. Hence, even for a real matrix `a` the eigenvalues
//! and eigenvectors returned are of the corresponding complex type.
//!
//! To just compute the eigenvalues of the matrix `a` we can use the following instead.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::{dot, diag};
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! # use rlst::base_types::c64;
//! # use rlst::dense::linalg::lapack::eigenvalue_decomposition::EigMode;
//! # use rlst::EigenvalueDecomposition;
//! # use rlst::Inverse;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! # a.fill_from_standard_normal(&mut rng);
//! # let (lam2, _, _) = a.eig(EigMode::BothEigenvectors).unwrap();
//! let lam = a.eigenvalues().unwrap();
//! # rlst::assert_array_relative_eq!(lam, lam2, 1E-10);
//! ```
//! With the parameter enum [EigMode](crate::dense::linalg::lapack::eigenvalue_decomposition::EigMode) one can control whether
//! to compute only eigenvalues, only the left eigenvectors, only the right eigenvectors, or all eigenvectors.
//!
//! # Schur Decomposition
//!
//! The Schur decomposition for a `n x n` matrix `A` is defined as `A = Q * R * Q^H`
//! with `Q` having orthogonal columns of unit length and `R` upper triangular. The diagonal
//! values of `R` contain the eigenvalues of `A`. The following example computes the Schur
//! decomposition. The Schur decomposition is contained in the same trait [EigenvalueDecomposition](crate::EigenvalueDecomposition)
//! as the eigenvalue decomposition.
//! ```
//! # extern crate lapack_src;
//! # extern crate blas_src;
//! # use rand::SeedableRng;
//! # use rand_chacha::ChaCha8Rng;
//! # use rlst::base_types::TransMode;
//! # use rlst::{diag, dot};
//! # let mut rng = ChaCha8Rng::seed_from_u64(0);
//! # use rlst::EvaluateObject;
//! # let mut a = rlst::rlst_dynamic_array!(f64, [5, 5]);
//! use rlst::EigenvalueDecomposition;
//! a.fill_from_standard_normal(&mut rng);
//! let (r, q) = a.schur().unwrap();
//! rlst::assert_array_relative_eq!(dot!(q.r(), r, q.r().conj().transpose().eval()), a, 1E-10);
//! ```
