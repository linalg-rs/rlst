//! An introduction to dense linear algebra with RLST.
//!
//! - [Basic concepts](#basic-concepts)
//! - [Element access](#element-access)
//! - [Stack based array allocations](#stack-based-array-allocations)
//! - [Arrays from memory slices](#arrays-from-memory-slices)
//! - [Strides](#strides)
//! - [References and ownership](#references-and-ownership)
//! - [Array slicing](#array-slicing)
//! - [Array iterators](#array-iterators)
//! - [Operations on arrays](#operations-on-arrays)
//! - [Matrix multiplication](#matrix-multiplication)
//!
//! # Basic concepts
//!
//! Let us start by defining a random `(3, 5, 2)` array.
//! ```
//! use rand::Rng;
//! let mut rng = rand::thread_rng();
//! let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! arr1.fill_from_equally_distributed(&mut rng);
//! ```
//!
//! Alternatively, you can also create the array with
//! ```
//! let shape = [3, 5, 2];
//! let arr = rlst::DynArray::<f64, 3>::from_shape(shape);
//! ````
//! Note that the macro `rlst_dynamic_array` relies on the compiler to automatically infer the
//! number of dimensions from the type of the provided shape. To provide the dimension explicitly
//! use [DynArray::from_shape](crate::DynArray::from_shape) directly.
//!
//! To create a memory aligned array you can give the alignment as a third parameter.
//! ```
//! let shape = [3, 5, 2];
//! // Create a cache aligned array
//! let arr = rlst::rlst_dynamic_array!(f64, shape | rlst::CACHE_ALIGNED);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % rlst::CACHE_ALIGNED, 0);
//! // Create a page aligned array
//! let arr = rlst::rlst_dynamic_array!(f64, shape | rlst::PAGE_ALIGNED);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % rlst::PAGE_ALIGNED, 0);
//! // Create an array with 64 bytes alignment
//! let arr = rlst::rlst_dynamic_array!(f64, shape | 64);
//! assert_eq!(arr.data().unwrap().as_ptr().addr() % 64, 0);
//! ```
//!
//! Any type that supports the [std::marker::Copy] and [std::default::Default] traits is supported
//! for arrays. Hence, the following is a valid construct.
//! ```
//! let mut arr = rlst::rlst_dynamic_array!([usize; 3], [3, 7]);
//! arr[[0, 1]] = [1, 2, 3];
//! ```
//! This defines an array of dimension `[3, 7]`, where each element is of type `[usize; 3]`.
//! We then set the element at position `[0, 1]` to have the value `[1, 2, 3]`.
//!
//!
//! Let us define a second `(3, 5, 2)` array and do some operations.
//! ```
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! arr2.fill_from_equally_distributed(&mut rng);
//!
//! let res = 3.0 * arr1 + 5.0 * arr2;
//! ```
//! The variable `res` is of a different type from `arr1` and `arr2`. It is an addition type that
//! represents the addition of two arrays. RLST internally uses an expression template arithmetic.
//! Array operations are only executed when requested by the user. We can evaluate the expression
//! into a new array using the following command.
//! ```
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! # let res = 3.0_f64 * arr1 + 5.0_f64 * arr2;
//! use rlst::EvaluateObject;
//! let output = res.eval();
//! ```
//! The required trait is [EvaluateObject](crate::EvaluateObject). Their is a quirk
//! in the type inference from Rust. Without explicitly annotating the constants `3.0`
//! and `5.0` as being [f64] Rust will not compile the statement as it cannot infer the
//! correct item type for the new array.
//!
//! There is an important principle behind this. Algebraic operations on arrays do not allocate new memory on
//!  the heap without the user explicitly saying so. New memory is first allocated when
//! the expression `3.0_f64 * arr1 + 5.0_f64 * arr2` is evaluated. In the background the evaluation
//! instantiate a single new array of the correct size and then goes through the expression in a single for-loop
//! to evaluate it componentwise.
//
//! The operation `3.0 * arr1 + 5.0 * arr2` takes ownership of `arr1` and `arr2`. If this is not desired
//! whe can create reference object to these arrays which behave like the original array but internally
//! store a reference to the original array. This is done with `arr1.r()` for non-mutable reference
//! objects and with `arr1.r_mut()` for mutable reference objects.
//! ```
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst::rlst_dynamic_array!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! let output = (3.0_f64 * arr1.r() + 5.0_f64 * arr2.r());
//! ```
//!
//! # Element access
//!
//! To get the value of an element of an array at an arbitrary position the
//! method [get_value](crate::RandomAccessByValue::get_value)
//! is provided. [get_value](crate::RandomAccessByValue::get_value) does not return a reference but an actual value. It can //
//! therefore always be used even when the array represents an expression rather than values in memory. If it is possible
//! to have a reference to an element (i.e. the array is associated
//! with elements in memory) then direct index notation is also possible. Both are shown below.
//!
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(f64, [3, 4, 2]);
//! arr.fill_from_seed_equally_distributed(0);
//! assert_eq!(arr.get_value([2, 3, 1]).unwrap(), arr[[2, 3, 1]]);
//! ```
//! The index notation `arr[[2, 3, 1]]` is just short for `arr.get([2, 3, 1]).unwrap()` and returns a reference.
//! The index notation `arr[[2, 3, 1]]` can also be used for a mutable reference, or equivalently use
//! `arr.get_mut([2, 3, 1]).unwrap()`. If the index does not exist the index notation returns an assertion error
//! while `get` and `get_mut` return `None`.
//!
//! # Stack based Array allocations
//!
//! RLST arrays can be allocated either on the heap or on the stack. Heap based allocations can be done dynamically at runtime.
//! Stack based allocations happens at compile time and the exact size of the array must be known. Stack based
//! arrays are of advantage for small fixed sizes when the overhead of allocating memory at runtime can impact performance.
//!
//! For heap-based allocation we have already seen the macro [rlst_dynamic_array](crate::rlst_dynamic_array) and the corresponding method [DynArray::from_shape](crate::DynArray::from_shape).
//! For stack based allocation the number of elements needs to be known at compile time. To make the allocation straight
//! forward the macro [rlst_static_array](crate::rlst_static_array). is provided. This works in the same way as
//! `rlst_dynamic_array`. However, the dimensions must be given explicitly as compile-time integers. The proc macro
//! then multiplies the dimensions and instantiates a compile time fixed size Rust array to hold the data. The following
//! shows an example.
//!
//! ```
//! # use rlst::EvaluateObject;
//! # let mut rng = rand::thread_rng();
//! let mut arr1 = rlst::rlst_static_array!(f64, [1, 3, 2]);
//! let mut arr2 = rlst::rlst_static_array!(f64, [1, 3, 2]);
//! arr1.fill_from_equally_distributed(&mut rng);
//! arr2.fill_from_equally_distributed(&mut rng);
//! let output = (arr1 + arr2).eval();
//! ```
//! What is interesting about this code is that all the required memory is allocated at compile time, including
//! the array `output`. The `eval` method sees that all arrays of the expression are stack allocated and
//! hence uses a compile time allocation on the stack for the output array. The decision about whether a
//! runtime or compile time allocation is performed, is decided as follows:
//! - If all arrays in the expression are heap based runtime allocated arrays, then `eval` creates a runtime
//!   array.
//! - If at least one of the arrays is a stack allocated compile time array then a stack-based allocation for
//!   `output` at compile time is performed.
//!
//! The rationale is that if at least one array in the expression is stack allocated, size information is
//! available at compile time so can be used to create the output. If every array in the expression is generated
//! at runtime then no compile time size information is available.
//!
//! The mechanism that makes this selection possible is fully implemented through traits and type mechanics.
//! So there is no runtime overhead.
//!
//! ## Arrays from memory slices
//!
//! One can create an array from a given memory slice. The following example shows how to do this.
//! ```
//! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let arr = rlst::SliceArray::<f64, 2>::from_shape(myvec.as_slice(), [2, 3]);
//! ```
//! For mutable slices the method [SliceArrayMut::from_shape](crate::SliceArrayMut::from_shape) exists.
//!
//! # Strides
//!
//! By default RLST uses a column major ordering, that is the left-most dimension is contiguous in memory.
//! This holds for memory storage order and also iterators over arrays. Any newly allocated array is automatically
//! using column-major order. However, other strides are possible. An array with a different stride can be initialized
//! with  [StridedDynArray::from_shape_and_stride](crate::StridedDynArray::from_shape_and_stride) or [StridedDynArray::row_major](crate::StridedDynArray::row_major).
//! to specifically crate an array in row-major format. Sometimes one needs to convert an array from colum-major to row-major format.
//! This can be done with [EvaluateRowMajorArray::eval_row_major](crate::EvaluateRowMajorArray::eval_row_major), which is implemented for all array types.
//!
//! All arrays with custom strides still iterate through in column-major order. This is to ensure that componentwise binary operations on two arrays are
//! possible for arbitrary stride arrays. The consequence is that iteration is a bit slower if an array has a custom stride. Also, the memory index calculation
//! is more involved, creating additional overhead. Where performance is crucial column-major arrays should always be preferred.
//!
//!
//!
//!
//! # References and ownership
//!
//! Many operations in RLST take ownership of their arguments, e.g. in the expression `let out = a + b` for arrays `a` and `b` the object
//! `out` takes ownership of `a` and `b`. To avoid this one can instead say `let out = a.r() + b.r()`. The method [r](crate::dense::array::Array::r)
//! returns an array that is implemented through an [ArrayRef](crate::dense::array::reference::ArrayRef) struct. An `ArrayRef` is a simple container
//! that holds a Rust reference to an actual array. This allows us to hand over references to arrays to functions that expect to take ownership.
//! For mutable arrays a corresponding function exists in the form of [r_mut](crate::dense::array::Array::r_mut). These reference
//! arrays behave exactly like the original arrays and implement the same traits. They can be transparantly used instead of
//! the original arrays.
//!
//! # Array slicing
//!
//! RLST allows the slicing of arrays across a given dimension. Given the following array.
//! ```
//! let arr = rlst::rlst_dynamic_array!(f64, [5, 3, 2]);
//! ```
//! we can slice at the first index of the second dimension as follows.
//! ```
//! # let arr = rlst::rlst_dynamic_array!(f64, [5, 3, 2]);
//! let slice = arr.r().slice::<2>(1, 0);
//! ```
//! This creates a two dimensional array of dimension `(5, 2)`. Any index `[a, b]` into this array
//! is mapped to an index `[a, 0, b]` of the original array. A slice is mutable if the original array was mutable.
//! A frequent example is access to a single column or row in a given matrix.
//! ```
//! let arr = rlst::rlst_dynamic_array!(f64, [4, 6]);
//! // This gives the third column
//! let slice = arr.r().slice::<1>(1, 2);
//! // This gives the fourth row
//! let slice = arr.r().slice::<1>(0, 3);
//! ```
//! The annotation with the dimension parameter is necessary if the compiler cannot automatically infer from
//! subsequent operations what the correct number of dimensions after slicing is.
//!
//! # Array iterators
//!
//! The default iterator returns elements of an array in column-major order independent of the underlying stride.
//! The following example demonstrates the default iterator.
//! ```
//! let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! for elem in arr.iter_mut() {
//!     *elem = 2.0;
//! }
//!
//! ```
//! The possible iterators are `iter_value` for an iterator that returns array values, `iter_ref` for an iterator
//! that returns references, and `iter_mut` for a mutable iterator. For example, the array `out = a + b` as the result
//! of the sum of the arrays `a` and `b` only supports `iter_value` with the componentwise addition of the elements
//! of `a` and `b` are performed as part of the iteration.
//!
//! It is possible to obtain the multi-index together with the iterator as is shown in the following example.
//! ```
//! # let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! # arr.fill_from_seed_equally_distributed(0);
//! use rlst::AsMultiIndex;
//! let shape = arr.shape();
//! for (multi_index, elem) in arr.iter_value().enumerate().multi_index(shape) {
//!     assert_eq!(elem, arr[multi_index]);
//! }
//! ```
//! In addition to the above iterators the following special iterators are defined.
//! - [col_iter](crate::dense::array::Array::col_iter) - for 2-dimensional arrays to iterate through the columns.
//! - [row_iter](crate::dense::array::Array::row_iter) - for 2-dimensional arrays to iterate through the rows.
//! - [diag_iter_value](crate::dense::array::Array::diag_iter_value) - for n-dimensional arrays to iterate through the diagonal by value.
//! - [diag_iter_ref](crate::dense::array::Array::diag_iter_ref) - for n-dimensional arrays to iterate through the diagonal by reference.
//! - [diag_iter_mut](crate::dense::array::Array::diag_iter_mut) - for n-dimensional arrays to iterate through the diagonal mutably.
//!
//!
//! # Operations on arrays
//!
//! RLST uses a system of static lazy evaluation for array operations. Consider the following example.
//! ```
//! # let mut rng = rand::thread_rng();
//! let mut arr = rlst::rlst_dynamic_array!(f64, [2, 3]);
//! arr.fill_from_equally_distributed(&mut rng);
//! let res = 5.0 * arr.r();
//! ```
//! The array `res` is not connected with a memory region. It internally takes ownership
//! of a reference into `arr`. The scalar multiplication with `5.0` is performed whenever an element if `res` is requested,
//! either by an iterator or by direct random access. Hence, the operation `let res = 5.0 * arr.r()` has no cost itself.
//! But calling `res.get_value([0, 0])` would multiply the first element of `arr` by `5.0` and return the result of that
//! operation. To get a new array in which every element was multiplied by `5.0` simply call `let out = res.eval()`.
//!
//! RLST supports all the usual algebraic unary and binary operations on arrays. Results are stored as expressions and only evaluated
//! on element access or evaluation into a new array. In addition to standard algebraic operations the following special functions are
//! implemented: `conj, abs, square, abs_square, sqrt, exp, ln, recip, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh`.
//!
//! Hence, we can form an expression of the form `let out = (5.0 * a.sqrt() + b.exp()).eval()`. To evaluate the expression only a single iteration through
//! the arrays `a` and `b` is performed.
//!
//! Below is a list of further operations supported through the expression system in RLST.
//! - [cast](crate::dense::array::Array::cast): Cast from one type to another using [num::cast::cast].
//! - [coerce_dim](crate::dense::array::Array::coerce_dim): Coerce the dimension of an array from a generic parameter to a concrete integer.
//! - [mul_add](crate::traits::MulAdd): Multiply an array with a scalar and add to another array. This uses a componentwise `mul_add` that
//!   is usually converted into a single `mul_add` cpu instruction instead of a separate multiplication and addition.
//! - [reverse_axis](crate::dense::array::Array::reverse_axis): Reverse the elements in an array along a single axis.
//! - [transpose](crate::dense::array::Array::transpose): Reverse the order of the axes of an array.
//! - [with_eval_type](crate::dense::array::Array::with_container_type): Force the use of either a dynamic or static container for array evaluation.
//! - [unary_op](crate::dense::array::Array::unary_op): Apply an arbitrary unary function componentwise to an array.
//! - [into_type](crate::dense::array::Array::into_type): Use [std::convert::Into] to convert into another type.
//!
//! # Matrix multiplication
//!
//! RLST uses its BLAS interface to perform matrix multiplication.
//! To use matrix products make sure to have a BLAS/Lapack provider as explained [here](crate::doc::getting_started). Furthermore,
//! BLAS requires the data stored in memory using column-major order. Hence, array expressions are not supported and need to first be
//! evaluated.
//!
//! ```
//! # extern crate blas_src;
//! let arr1 = rlst::rlst_dynamic_array!(f64, [4, 5]);
//! let arr2 = rlst::rlst_dynamic_array!(f64, [5, 3]);
//! let res = rlst::dot!(arr1.r(), arr2.r());
//! ```
//! This multiplies the matrices `arr1` and `arr2` into a new matrix `res`. The macro [crate::dot] initializes a new array and multiplies
//! `arr1` and `arr2` into it. `dot` supports the multiplication of multiple arrays. Hence, an expression of the form `dot!(a1, a2, a3, a4)`
//! is possible. Arrays are multiplied with `dot` from right to left. For combined array multiplication/addition of the form `Y -> alpha * A B + beta * Y`
//! the [apply](crate::AsMatrixApply::apply) is implemented for all array types that support BLAS operations.
//!
//
//
//
// ! # Pivoted QR Decomposition
// !
// ! The pivted QR decomposition of a matrix `A` is given as `AP=QR`, where `P` is a permutation matrix, `R` is upper triangular, and
// ! `Q` has orthogonal columns. For details see the documentation of [MatrixQrDecomposition](crate::dense::linalg::qr::MatrixQr).
// !
// ! The pivoted QR decomposition of a matrix can be computed as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut r_mat = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut p_mat = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut q_mat = rlst_dynamic_array2!(f64, [8, 5]);
//  ! arr.fill_from_equally_distributed(&mut rand);
// ! let qr = arr.into_qr_alloc().expect("QR Decomposition failed");
// ! qr.get_r(r_mat.r_mut());
// ! qr.get_q_alloc(q_mat.r_mut());
// ! qr.get_p(p_mat.r_mut());
// ! ````
// ! The content of `arr` is overwritten with the QR decomposition. The method [get_q_alloc](crate::dense::linalg::qr::QrDecomposition::get_q_alloc)
// ! needs to allocate additional temporary memory on the heap. This is why it is annoted with `_alloc`.
// !
// ! # Singular value decomposition
// !
// ! To compute the singular values of a two-dimensional array `arr` use
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut singvals = rlst_dynamic_array1!(f64, [5]);
// ! arr.into_singular_values_alloc(singvals.data_mut()).unwrap();
// ! ```
// ! This computes the singular values of `arr` into `singvals`. The method [into_singular_values_alloc](crate::MatrixSvd::into_singular_values_alloc)
// ! needs to allocate temporary memory on the heap. This is why it has the ending `_alloc`. To compute the whole
// ! singular value decomposition use the method [into_svd_alloc](crate::MatrixSvd::into_svd_alloc).
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut u = rlst_dynamic_array2!(f64, [8, 5]);
// ! let mut vt = rlst_dynamic_array2!(f64, [5, 5]);
// ! let mut sigma = rlst_dynamic_array1!(f64, [5]);
// ! arr.into_svd_alloc(u.r_mut(), vt.r_mut(), sigma.data_mut(), SvdMode::Reduced).unwrap();
// ! ```
// ! To compute the full SVD use the parameter [SvdMode::Full](crate::SvdMode::Full).
// !
// ! # Other vector functions
// !
// ! The following other functions for arrays with a single dimension (i.e. vectors) are provided.
// ! - [inner](crate::Array::inner): Compute the inner product with another vector.
// ! - [norm_1](crate::Array::norm_1): Compute the 1-norm of a vector.
// ! - [norm_2](crate::Array::norm_2): Compute the 2-norm of a vector.
// ! - [norm_inf](crate::Array::norm_inf): Compute the inf-norm of a vector.
// ! - [len](crate::Array::len): Convenience function to return the length of a vector.
// ! - [cross](crate::Array::cross): Compute the cross product with another vector.
// !
// ! # Other matrix functions
// !
// ! The following other functions for arrays with two dimensions (i.e. matrices) are provided.
// ! - [norm_1](crate::Array::norm_1): Compute the 1-norm of a matrix.
// ! - [norm_2_alloc](crate::Array::norm_2): Compute the 2-norm of a matrix. This method allocates temporary memory on the heap.
// ! - [norm_fro](crate::Array::norm_fro): Compute the Frobenius norm of a matrix.
// ! - [norm_inf](crate::Array::norm_inf): Compute the inf-norm of a matrix.
// ! - [into_det](crate::Array::into_det): Compute the determinant of a matrix.
// ! - [into_pseudo_inverse_alloc](crate::MatrixPseudoInverse::into_pseudo_inverse_alloc) and [into_pseudo_inverse_resize_alloc](crate::MatrixPseudoInverse::into_pseudo_inverse_resize_alloc) compute the pseudoinverse of a matrix.
