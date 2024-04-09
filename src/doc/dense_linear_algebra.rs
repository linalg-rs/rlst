//! An introduction to dense linear algebra with RLST.
//!
//!
//! # Basic operations
//!
//! Let us start by defining a random `(3, 5, 2)` array.
//! ```
//! use rlst::prelude::*;
//! use rand::Rng;
//! let mut rng = rand::thread_rng();
//! let mut arr1 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! arr1.fill_from_equally_distributed(&mut rng);
//! ```
//! [rlst_dynamic_array3](crate::rlst_dynamic_array3) is a macro
//! that initialises a new array on the heap filled with zeros.
//! The following types are supported: [f32], [f64], [c32](crate::dense::types::c32), [c64](crate::dense::types::c64).
//! Let us define a second `(3, 5, 2)` array and do some operations.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! let mut arr2 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! arr2.fill_from_equally_distributed(&mut rng);
//!
//! let res = 3.0 * arr1 + 5.0 * arr2;
//! ```
//! The variable `res` is of a different type from `arr1` and `arr2`. It is an addition type that
//! represents the addition of two arrays. RLST internally uses an expression template arithmetic.
//! Array operations are only executed when requested by the user. We can evaluate the expression
//! into a new array using the following command.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! # let res = 3.0 * arr1 + 5.0 * arr2;
//! let mut output = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! output.fill_from(res);
//! ```
//! We can write the whole operation shorter as follows.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! let output = empty_array().fill_from_resize(3.0 * arr1 + 5.0 * arr2);
//! ```
//! There is an important principle behind this. All operations in RLST that allocate new memory on the heap are explicit.
//! Simply writing `let res = 3.0 * arr1 + 5.0 * arr2` does not allocate new memory on the heap. If we want to evaluate
//! this operation into a new array we have to create this ourself. To make this easier the function `empty_arry` exists.
//! It creates a new array of zero size. Many RLST routines have variants with the additional marker `_resize`. These can
//! resize an existing array when required. So the command `let output = empty_array().fill_from_resize(3.0 * arr1 + 5.0 * arr2);`
//! creates a new empty array and then resizes it to fill with the result of the array operation.
//!
//! The operation `3.0 * arr1 + 5.0 * arr2` takes ownership of the variables `arr1` and `arr2`. Most operations in RLST by default
//! take ownership. If this is not desired one can instead use a `view` object.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! # let mut arr1 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr1.fill_from_equally_distributed(&mut rng);
//! # let mut arr2 = rlst_dynamic_array3!(f64, [3, 5, 2]);
//! # arr2.fill_from_equally_distributed(&mut rng);
//! let output = empty_array().fill_from_resize(3.0 * arr1.view() + 5.0 * arr2.view());
//! ```
//! The method `arr1.view()` creates a `view` object that stores a reference to `arr1`. Ownership is now taken of the `view` object and
//! not of `arr1`. Ownership rules of Rust naturally extend to `view` objects since they are container that store Rust references. A mutable
//! view is created through `arr1.view_mut()`.
//!
//! # Array allocations
//!
//! RLST arrays can be allocated either on the heap or on the stack. Heap based allocations can be done dynamically at runtime.
//! Stack based allocations happens at compile time and the exact size of the array must be known. Stack based arrays are of advantage
//! for small fixed sizes when the overhead of allocating memory at runtime can impact performance.
//!
//! ## Heap based array allocations
//!
//! A heap based allocation is done by the macro `rlst_dynamic_arrayX`, where `X` is between 1 and 5 and denotes the number of axes of the
//! array. Higher axes numbers are possible. But we do not provide convenience macros for these cases. To allocate an array with two axes
//! that can hold double precision numbers we use
//! ```
//! # use rlst::prelude::*;
//! let mut arr = rlst_dynamic_array2!(f64, [5, 4]);
//! ```
//! This is a short form for the following longer initialisation.
//! ```
//! # use rlst::prelude::*;
//! let mut arr = DynamicArray::<f64, 2>::from_shape([5, 4]);
//! ````    
//!
//! ## Stack based array allocations.
//!
//! Allocating an array on the stack works quite similarly.
//! ```
//! # use rlst::prelude::*;
//! let mut arr = rlst_static_array!(f64, 5, 4);
//! ```
//! Internally, a heap based RLST array uses a Rust [Vec] while a Stack based array uses a Rust [array] type to store data.
//! Both, heap based and stack based arrays implement the same traits to provide the same functionality. However, stack based
//! arrays cannot be resized.
//!
//! ## Arrays from memory slices
//!
//! One can create an array from a given memory slice. The following example shows how to do this.
//! ```
//! # use rlst::prelude::*;
//! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let arr = rlst_array_from_slice2!(myvec.as_slice(), [2, 3]);
//! ```
//! To initialise the array from the slice with a non-standard slice one can provide a slice array explicitly. The following creates
//! an array from the slice using a row-major memory ordering.
//! ```
//! # use rlst::prelude::*;
//! let myvec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let stride = [3, 1];
//! let arr = rlst_array_from_slice2!(myvec.as_slice(), [2, 3], stride);
//! ```
//!
//!
//! ## Strides
//!
//! The stride of an array denotes how elements are mapped from n-dimensions to 1-dimensional memory locations. Let the stride
//! be [s1, s2, s3, ..] then an index (a1, a2, a3, ...) is mapped to a 1-dimensional index through `ind = a1 * s1 + a2 * s2 + a3 * s3 + ...`.
//! The default stride in RLST is column-major. This means that the first dimension is consecutive in memory. For heap based arrays
//! a non-default stride can be provided
//! ```
//! # use rlst::prelude::*;
//! let stride = [4, 1];
//! let mut arr = DynamicArray::<f64, 2>::from_shape_with_stride([5, 4], stride);
//! ```
//! The above array uses a row-major stride, meaning consecutive elements in a row are consecutive in memory.
//! RLST supports arbitrary strides. However, operations that rely on extenal BLAS or Lapack interfaces require a column-major ordering.
//!
//! ## Views and mutable views
//!
//! Most operations in RLST take ownership of their arguments. If we want to avoid this we can create `view` objects. A borrowed view is
//! created as follows.
//! ```
//! # use rlst::prelude::*;
//! let mut arr = rlst_dynamic_array2!(f64, [5, 4]);
//! let view = arr.view();
//! ```
//! To create a mutable view use instead `arr.view_mut()`. Views are just contianer that hold references to the original array. They implement
//! all traits that also arrays implement and can be used interchangeably. Hence, of a RLST function takes ownership of an array one can provide
//! instead a `view` object and the function takes ownership of the view and not of the original array.
//!
//! ## Operations on arrays
//! ### Static lazy evaluation
//! RLST uses a system of static lazy evaluation for array operations. Consider the following example.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
//! arr.fill_from_equally_distributed(&mut rng);
//! let res = 5.0 * arr.view();
//! ```
//! The array `res` is not connected with a memory region. It internally takes ownership of a view into `arr` and every random access to that
//! view is multipltakes ownership of a view into `arr` and every random access to that
//! view is multiplied by `5.0`. The `res` array implements also all traits that other arrays implement and can be used like any other array.
//! The only exception is raw access to its memory. Since it does not have its own memory it does not implement the [RawAccess](crate::RawAccess) trait.
//! To evaluate the operation `5.0 * arr.view()` one creates a new array and fills this array with values as follows.
//! ```
//! # use rlst::prelude::*;
//! # let mut rng = rand::thread_rng();
//! let mut arr = rlst_dynamic_array2!(f64, [2, 3]);
//! let mut res = rlst_dynamic_array2!(f64, [2, 3]);
//! arr.fill_from_equally_distributed(&mut rng);
//! res.fill_from(5.0 * arr.view());
//! ```
//! The [fill_from](crate::dense::array::Array::fill_from) method iterates through `5.0 * arr.view()` and fills the array `res` with the returned values.
//! The advantage of this approach is that more complex operations such as `res.fill_from(5.0 * arr1.view() + 3.0 * arr2.view())` only require a single
//! loop through the data and create no temporary objects. RLST provides [various different methods](#evaluating-arrays-into-other-arrays) to evaluate an
//! array into another array.
//!
//! # Evaluating arrays into other arrays
//!
//! To simply fill an array with the values of an other array the method [fill_from](crate::dense::array::Array::fill_from) is provided. See the following example.
//! ```
//! # use rlst::prelude::*;
//! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
//! # let mut arr2 = rlst_dynamic_array2!(f64, [2, 3]);
//! # arr2.fill_from(arr1.view());
//! ```
//! The [fill_from](crate::dense::array::Array::fill_from) method takes ownership of its arguments. If this is not desired a `view` should passed into the method as
//! in the example above. The [fill_from](crate::dense::array::Array::fill_from) method assumes that both arrays have the same shape and type. However, they need not
//! have the same stride. A variant that allows resizing of the target array is given by [fill_from_resize](crate::dense::array::Array::fill_from_resize). With this method
//! a simple heap allocated copy of an array can be created through
//! ```
//! # use rlst::prelude::*;
//! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
//! let arr2 = empty_array().fill_from_resize(arr1.view());
//! ```
//! Instead of `fill_from` the method [sum_into](crate::dense::array::Array::sum_into) can be used to sum the values of one array into another array. This method has no `resize` variant.
//!
//! Both `fill_from` and `sum_into` may not be automatically SIMD vectorized by the compiler. The compiler has no assumptions on how the elements of the two arrays are accessed in memory.
//! To solve this problem a chunked evaluation is possible. Consider the following example.
//! ```
//! # use rlst::prelude::*;
//! # let arr1 = rlst_dynamic_array2!(f64, [2, 3]);
//! # let mut arr2 = rlst_dynamic_array2!(f64, [2, 3]);
//! # arr2.fill_from_chunked::<_, 16>(5.0 * arr1.view());
//! ```
//! The chunked evaluation passes through `arr1.view()` in chunks of 16, copying 16 elements at a time into a buffer and then multiplying each element in the buffer by `5.0`. This allows effective
//! SIMD evaluation but comes at a price of an additional copy operation. The buffer is not heap allocated but lives on the stack and its size is determined as const generic parameter at compile time.
//! Good sizes are multiples of the SIMD vector length of a CPU. The number of elements in an array does not need to be an exact multiple of the chunk size. This case is handled correctly in RLST.
//! In summary, the following methods provide evaluation of an array into another array.
//! - [fill_from](crate::dense::array::Array::fill_from)
//! - [fill_from_chunked](crate::dense::array::Array::fill_from_chunked)
//! - [fill_from_resize](crate::dense::array::Array::fill_from_resize)
//! - [fill_from_chunked_resize](crate::dense::array::Array::fill_from_chunked_resize)
//! - [sum_into](crate::dense::array::Array::sum_into)
//! - [sum_into_chunked](crate::dense::array::Array::sum_into_chunked)
