//! An introduction to dense linear algebra with RLST.
//!
//! - [Basic concepts](#basic-concepts)
//! - [Element access](#element-access)
//! - [Array allocations](#array-allocations)
//!    - [Heap based array allocations](#heap-based-array-allocations)
//!    - [Stack based array allocations](#stack-based-array-allocations)
//!    - [Arrays from memory slices](#arrays-from-memory-slices)
//! - [Strides](#strides)
//! - [Views and mutable views](#views-and-mutable-views)
//! - [Array slicing](#array-slicing)
//! - [Array iterators](#array-iterators)
//! - [Operations on arrays](#operations-on-arrays)
//!    - [Static lazy evaluation](#static-lazy-evaluation)
//!    - [Multiplication with a scalar](#multiplication-with-a-scalar)
//!    - [Negation of an array](#negation-of-an-array)
//!    - [Addition and subtraction of arrays](#addition-and-subtraction-of-arrays)
//!    - [Componentwise multiplication and division](#componentwise-multiplication-and-division)
//!    - [Transposition of arrays](#transposition-of-arrays)
//!    - [Conversion to complex type](#conversion-to-complex-type)
//!    - [Conjugation of an array](#conjugation-of-an-array)
//! - [Evaluating arrays into other arrays](#evaluating-arrays-into-other-arrays)
//! - [Matrix multiplication](#matrix-multiplication)
//! - [LU Decomposition and solving linear systems of equations](#lu-decomposition-and-solving-linear-systems-of-equations)
//! - [Pivoted QR Decomposition](#pivoted-qr-decomposition)
//! - [Singular value decomposition](#singular-value-decomposition)
//! - [Other vector functions](#other-vector-functions)
//! - [Other matrix functions](#other-matrix-functions)
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
//! Note that the macro `rlst_dynamic_array` only allows array literals as a second parameter.
//! This is because the macro infers from the number of elements in the array literal how many
//! dimensions there are. If the shape of the array is provided in the form of a variable (e.g. `shape` as above)
//! the function [DynArray::from_shape](crate::DynArray::from_shape) needs to be used and the number of
//! dimensions explicitly provided as in `DynArray::<f64, 3>::from_shape`. The reason is that Rust does
//! not yet allow to infer the value of const generic arguments in types.
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
//! To get the value of an element of an array at an arbitrary position the method [get_value](crate::RandomAccessByValue::get_value)
// ! is provided. [get_value](crate::RandomAccessByValue::get_value) does not return a reference but an actual value. It can therefore always be used
// ! even when the array represents a complicated expression. If it is possible to have a reference to an element (i.e. the array is associated
// ! with elements in memory) then direct index notation is also possible. Both are shown below.
// !
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = rlst_dynamic_array3!(f64, [3, 4, 2]);
// ! arr.fill_from_seed_equally_distributed(0);
// ! assert_eq!(arr.get_value([2, 3, 1]).unwrap(), arr[[2, 3, 1]]);
// ! ```
// ! To get a reference to an element use the method [get](crate::RandomAccessByRef::get). A mutable variant is available as [get_mut](crate::RandomAccessMut::get_mut).
// ! These methods by default perform bounds checking. Unsafe variants without bounds checking are available in the traits [UnsafeRandomAccessByRef](crate::UnsafeRandomAccessByRef),
// ! [UnsafeRandomAccessByValue](crate::UnsafeRandomAccessByValue), and [UnsafeRandomAccessMut](crate::UnsafeRandomAccessMut).
// !
// ! # Array allocations
// !
// ! RLST arrays can be allocated either on the heap or on the stack. Heap based allocations can be done dynamically at runtime.
// ! Stack based allocations happens at compile time and the exact size of the array must be known. Stack based arrays are of advantage
// ! for small fixed sizes when the overhead of allocating memory at runtime can impact performance.
// !
// ! ## Heap based array allocations
// !
// ! A heap based allocation is done by the macro `rlst_dynamic_arrayX`, where `X` is between 1 and 5 and denotes the number of axes of the
// ! array. Higher axes numbers are possible. But we do not provide convenience macros for these cases. To allocate an array with two axes
// ! that can hold double precision numbers we use
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = rlst_dynamic_array2!(f64, [5, 4]);
// ! ```
// ! This is a short form for the following longer initialisation.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = DynamicArray::<f64, 2>::from_shape([5, 4]);
// ! ````
// !
// ! ## Stack based array allocations.
// !
// ! Allocating an array on the stack works quite similarly.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = rlst_static_array!(f64, 5, 4);
// ! ```
// ! Internally, a heap based RLST array uses a Rust [Vec] while a Stack based array uses a Rust [array] type to store data.
// ! Both, heap based and stack based arrays implement the same traits to provide the same functionality. However, stack based
// ! arrays cannot be resized.
// !
// ! ## Arrays from memory slices
// !
// ! One can create an array from a given memory slice. The following example shows how to do this.
// ! ```
// ! # use rlst::prelude::*;
// ! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
// ! let arr = rlst_array_from_slice2!(myvec.as_slice(), [2, 3]);
// ! ```
// ! To initialise the array from the slice with a non-standard slice one can provide a slice array explicitly. The following creates
// ! an array from the slice using a row-major memory ordering.
// ! ```
// ! # use rlst::prelude::*;
// ! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
// ! let stride = [3, 1];
// ! let arr = rlst_array_from_slice2!(myvec.as_slice(), [2, 3], stride);
// ! ```
// !
// ! # Strides
// !
// !
// ! # Views and mutable views
// !
// ! Most operations in RLST take ownership of their arguments. If we want to avoid this we can create `view` objects. A borrowed view is
// ! created as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = rlst_dynamic_array2!(f64, [5, 4]);
// ! let view = arr.r();
// ! ```
// ! To create a mutable view use instead `arr.r_mut()`. Views are just contianer that hold references to the original array. They implement
// ! all traits that also arrays implement and can be used interchangeably. Hence, of a RLST function takes ownership of an array one can provide
// ! instead a `view` object and the function takes ownership of the view and not of the original array.
// !
// ! # Array slicing
// !
// ! RLST allows the slicing of arrays across a given dimension. Given the following array.
// ! ```
// ! # use rlst::prelude::*;
// ! let arr = rlst_dynamic_array3!(f64, [5, 3, 2]);
// ! ```
// ! we can slice at the first index of the second dimension as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr = rlst_dynamic_array3!(f64, [5, 3, 2]);
// ! let slice = arr.r().slice(1, 0);
// ! ```
// ! This creates a two dimensional array of dimension `(5, 2)`. Any index `[a, b]` into this array
// ! is mapped to an index `[a, 0, b]` of the original array. A slice is mutable if the original array or view was mutable.
// ! A frequent example is access to a single column or row in a given matrix.
// ! ```
// ! # use rlst::prelude::*;
// ! let arr = rlst_dynamic_array2!(f64, [4, 6]);
// ! // This gives the third column
// ! let slice = arr.r().slice(1, 2);
// ! // This gives the fourth row
// ! let slice = arr.r().slice(0, 3);
// ! ```
// !
// ! # Array iterators
// !
// ! The default iterator returns elements of an array in column-major order independent of the underlying stride.
// ! The following example demonstrates the default iterator.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! for elem in arr.iter_mut() {
// !     *elem = 2.0;
// ! }
// ! ```
// ! This fills all elements of an array with the value `2.0`. It is often useful to get the multi index of an element together
// ! with the element itself. This can be achieved as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! # arr.fill_from_seed_equally_distributed(0);
// ! let shape = arr.shape();
// ! for (multi_index, elem) in arr.iter().enumerate().multi_index(shape) {
// !     assert_eq!(elem, arr[multi_index]);
// ! }
// ! ```
// ! The following iterators are provided by default.
// ! - [iter](crate::DefaultIterator): Default column-major iterator.
// ! - [iter_mut](crate::DefaultIteratorMut::iter_mut): Default mutable column-major iterator.
// ! - [row_iter](crate::Array::row_iter): Iterator over rows of a two-dimensional array.
// ! - [row_iter_mut](crate::Array::row_iter_mut): Mutable iterator over rows of a two-dimensional array.
// ! - [col_iter](crate::Array::col_iter): Iterator over columns of a two-dimensional array.
// ! - [col_iter_mut](crate::Array::col_iter_mut): Mutable iterator over columns of a two-dimensional array.
// !
// !
// ! ## Operations on arrays
// ! ### Static lazy evaluation
// ! RLST uses a system of static lazy evaluation for array operations. Consider the following example.
// ! ```
// ! # use rlst::prelude::*;
// ! # let mut rng = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! arr.fill_from_equally_distributed(&mut rng);
// ! let res = 5.0 * arr.r();
// ! ```
// ! The array `res` is not connected with a memory region. It internally takes ownership of a view into `arr` and every random access to that
// ! view is multipltakes ownership of a view into `arr` and every random access to that
// ! view is multiplied by `5.0`. The `res` array implements also all traits that other arrays implement and can be used like any other array.
// ! The only exception is raw access to its memory. Since it does not have its own memory it does not implement the [RawAccess](crate::RawAccess) trait.
// ! To evaluate the operation `5.0 * arr.r()` one creates a new array and fills this array with values as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let mut rng = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let mut res = rlst_dynamic_array2!(f64, [2, 3]);
// ! arr.fill_from_equally_distributed(&mut rng);
// ! res.fill_from(5.0 * arr.r());
// ! ```
// ! The [fill_from](crate::dense::array::Array::fill_from) method iterates through `5.0 * arr.r()` and fills the array `res` with the returned values.
// ! The advantage of this approach is that more complex operations such as `res.fill_from(5.0 * arr1.r() + 3.0 * arr2.r())` only require a single
// ! loop through the data and create no temporary objects. RLST provides [various different methods](#evaluating-arrays-into-other-arrays) to evaluate an
// ! array into another array.
// !
// ! ## Multiplication with a scalar
// !
// ! The following options are available to multiply an array with a scalar.
// ! ```
// ! # use rlst::prelude::*;
// ! # let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! arr.scale_inplace(5.0);
// ! let res1 = 5.0 * arr.r();
// ! let res2 = arr.r().scalar_mul(5.0);
// ! ```
// ! The method [scale_in_place](crate::Array::scale_inplace) immediately scales in place all elements of the array by the given scalar. The commands
// ! `5.0 * arr` and `arr.r().scalar_mul(5.0)` are identical. They both return a new lazy evaluation object without performing the scalar multiplication.
// !
// ! ## Negation of an array
// !
// ! To negate the elements of an array use one of
// ! ```
// ! # use rlst::prelude::*;
// ! # let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res1 = arr.r().neg();
// ! let res2 = -1.0 * arr.r();
// ! ```
// ! Both operations are identical and return a lazy evaluation object.
// !
// ! ## Addition and subtraction of arrays
// !
// ! To add/subtract arrays use one of
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
// ! # let arr2 = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res1 = arr1.r() + arr2.r();
// ! let res2 = arr1.r().add(arr2.r());
// ! let res3 = arr1.r() - arr2.r();
// ! let res4 = arr1.r().sub(arr2.r());
// ! ```
// ! The add, respectively subtraction operations, are implemented in the same way and return a lazy evaluation object that
// ! represents addition/subtraction of the arrays.
// !
// ! ## Componentwise multiplication and division
// !
// ! Componentwise multiplication and division of two arrays is performed as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
// ! # let arr2 = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res1 = arr1.r() * arr2.r();
// ! let res2 = arr1.r() / arr2.r();
// ! ```
// ! Both operations use lazy evaluation.
// !
// ! ## Transposition of arrays
// !
// ! Lazy transposition of an array is performed as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res = arr.r().transpose();
// ! ```
// ! For general n-dimensional arrays transposition reverses the order of axes. More general axes permutations are available
// ! through the [permute_axes](crate::Array::permute_axes) method. Note that this method only performs transposition. To take the
// ! complex conjugate transpose use
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res = arr.r().conj().transpose();
// ! ```
// !
// ! ## Conversion to complex type
// !
// ! Sometimes it is useful to convert the type of an array to the corresponding complex type. This is done lazily as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res = arr.r().to_complex();
// ! ```
// !
// ! ## Conjugation of an array
// !
// ! Lazy complex conjugation of an array is done as below.
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr = rlst_dynamic_array2!(f64, [2, 3]);
// ! let res = arr.r().conj();
// ! ```
// !
// ! # Evaluating arrays into other arrays
// !
// ! To simply fill an array with the values of an other array the method [fill_from](crate::dense::array::Array::fill_from) is provided. See the following example.
// ! ```
// ! # use rlst::prelude::*;
// ! let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
// ! let mut arr2 = rlst_dynamic_array2!(f64, [2, 3]);
// ! arr2.fill_from(arr1.r());
// ! ```
// ! The [fill_from](crate::dense::array::Array::fill_from) method takes ownership of its arguments. If this is not desired a `view` should passed into the method as
// ! in the example above. The [fill_from](crate::dense::array::Array::fill_from) method assumes that both arrays have the same shape and type. However, they need not
// ! have the same stride. A variant that allows resizing of the target array is given by [fill_from_resize](crate::dense::array::Array::fill_from_resize). With this method
// ! a simple heap allocated copy of an array can be created through
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
// ! let arr2 = empty_array().fill_from_resize(arr1.r());
// ! ```
// ! Instead of `fill_from` the method [sum_into](crate::dense::array::Array::sum_into) can be used to sum the values of one array into another array. This method has no `resize` variant.
// !
// ! Both `fill_from` and `sum_into` may not be automatically SIMD vectorized by the compiler. The compiler has no assumptions on how the elements of the two arrays are accessed in memory.
// ! To solve this problem a chunked evaluation is possible. Consider the following example.
// ! ```
// ! # use rlst::prelude::*;
// ! let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
// ! let mut arr2 = rlst_dynamic_array2!(f64, [2, 3]);
// ! arr2.fill_from_chunked::<_, 16>(5.0 * arr1.r());
// ! ```
// ! The chunked evaluation passes through `arr1.r()` in chunks of 16, copying 16 elements at a time into a buffer and then multiplying each element in the buffer by `5.0`. This allows effective
// ! SIMD evaluation but comes at a price of an additional copy operation. The buffer is not heap allocated but lives on the stack and its size is determined as const generic parameter at compile time.
// ! Good sizes are multiples of the SIMD vector length of a CPU. The number of elements in an array does not need to be an exact multiple of the chunk size. This case is handled correctly in RLST.
// ! In summary, the following methods provide evaluation of an array into another array.
// ! - [fill_from](crate::dense::array::Array::fill_from)
// ! - [fill_from_chunked](crate::dense::array::Array::fill_from_chunked)
// ! - [fill_from_resize](crate::dense::array::Array::fill_from_resize)
// ! - [fill_from_chunked_resize](crate::dense::array::Array::fill_from_chunked_resize)
// ! - [sum_into](crate::dense::array::Array::sum_into)
// ! - [sum_into_chunked](crate::dense::array::Array::sum_into_chunked)
// !
// ! # Matrix multiplication
// !
// ! RLST uses its BLAS interface to perform matrix multiplication.
// ! To use matrix products make sure to have a BLAS/Lapack provider as explained [here](crate::doc::initialise_rlst). Furthermore,
// ! because of BLAS only column-major order is supported for matrix products. To convert a matrix to column-major order simply initialise
// ! a new matrix with column-major order (default stride) and evaluate the existing matrix into the new matrix.
// !
// ! The following operation is the most simple way of multiplying two matrices into a new matrix.
// !
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr1 = rlst_dynamic_array2!(f64, [4, 5]);
// ! # let arr2 = rlst_dynamic_array2!(f64, [5, 3]);
// ! let res = empty_array().simple_mult_into_resize(arr1.r(), arr2.r());
// ! ```
// ! This multiplies the matrices `arr1` and `arr2` into a new matrix `res`. The method [simple_mult_into_resize](crate::dense::traits::MultIntoResize::simple_mult_into_resize)
// ! resizes the result array if necessary. To multiply multiple arrays one can chain the `simple_mult_into_resize` method.
// !
// ! ```
// ! # use rlst::prelude::*;
// ! # let arr1 = rlst_dynamic_array2!(f64, [2, 2]);
// ! # let arr2 = rlst_dynamic_array2!(f64, [2, 2]);
// ! # let arr3 = rlst_dynamic_array2!(f64, [2, 2]);
// ! let res = empty_array().simple_mult_into_resize(
// !             empty_array().simple_mult_into_resize(arr1.r(), arr2.r()),
// !             arr3.r()
// !           );
// ! ```
// ! The rationale behind this interface is that the user may want to store results of the multiplication in stack allocated arrays.
// ! Hence, the matrix multiplication cannot implicitly allocate a new matrix on the heap. For multiplication into matrices
// ! that cannot be resied (e.g. stack allocate matrices) the method [simple_mult_into](crate::dense::traits::MultInto::simple_mult_into)
// ! should be used. This method does not rely on being able to resize an array. However, it panics if the result matrix does not have
// ! correct dimensions.
// !
// ! A more complete BLAS like interface is provided by the method [mult_into](crate::dense::traits::MultInto::mult_into), which performs
// ! `C = alpha * op(A) * op(B) + beta * C`, for matrices `A`, `B`, and `C`, where `op` is the identity, transpose, or conjugate transpose.
// !
// ! # LU Decomposition and solving linear systems of equations
// !
// ! Through its Lapack interface RLST can solve dense linear systems of equations. A linear system of equations is solved as follows.
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [4, 4]);
// ! let mut rhs = rlst_dynamic_array1!(f64, [4]);
// ! arr.fill_from_equally_distributed(&mut rand);
// ! let lu = LuDecomposition::<f64, _ >::new(arr).expect("LU Decomposition failed.");
// ! lu.solve_vec(TransMode::NoTrans, rhs.r_mut()).expect("LU solve failed.");
// ! ```
// ! The variable `rhs` is overwritten with the solution to the linear system of equations. For solving with multiple right-hand sides use the
// ! [solve_mat](crate::dense::linalg::lu::MatrixLuDecomposition::solve_mat). Note that the structure [LuDecomposition](crate::dense::linalg::lu::LuDecomposition)
// ! takes ownership of `arr` and overwrites the content of its memory with the LU Decomposition. To obtain the individual factors of the LU decomposition see
// ! the documentation of [MatrixLuDecomposition](crate::dense::linalg::lu::MatrixLuDecomposition).
// !
// ! If only a single solve is required and the LU factors need not be stored a shorter version is available as
// ! ```
// ! # use rlst::prelude::*;
// ! let mut rand = rand::thread_rng();
// ! let mut arr = rlst_dynamic_array2!(f64, [4, 4]);
// ! let mut rhs = rlst_dynamic_array1!(f64, [4]);
// ! arr.fill_from_equally_distributed(&mut rand);
// ! let sol = arr.into_solve_vec(TransMode::NoTrans, rhs).expect("LU solve failed.");
// ! ```
// ! The method [into_solve_vec](crate::dense::array::Array::into_solve_vec) takes ownership of `rhs` and the array `arr`. It returns ownership of `rhs` and
// ! overwrites it with the solution. For convenience `rhs` is then returned again as `sol`. The array `arr` is also overwritten with the LU factors.
// ! A corresponding routine [into_solve_mat](crate::dense::array::Array::into_solve_mat) is available.
// !
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
