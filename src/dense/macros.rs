//! The macros defined here make it easy to create new matrices and vectors.

/// Create a new two one dimensional heap allocated array.
///
/// A heap allocated array has a size that is determined at runtime.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// // Creates a (3) array with `f64` entries.
/// let arr = rlst_dynamic_array1!(f64, [3]);
/// ```
#[macro_export]
macro_rules! rlst_dynamic_array1 {
    ($scalar:ty, $shape:expr) => {{
        $crate::dense::array::DynamicArray::<$scalar, 1>::from_shape($shape)
    }};
}

/// Create a new two dimensional heap allocated array.
///
/// A heap allocated array has a size that is determined at runtime.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// // Creates a (3, 5) array with `f64` entries.
/// let arr = rlst_dynamic_array2!(f64, [3, 5]);
/// ```
#[macro_export]
macro_rules! rlst_dynamic_array2 {
    ($scalar:ty, $shape:expr) => {{
        $crate::dense::array::DynamicArray::<$scalar, 2>::from_shape($shape)
    }};
}

/// Create a new three dimensional heap allocated array.
///
/// A heap allocated array has a size that is determined at runtime.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// // Creates a (3, 5, 2) array with `f64` entries.
/// let arr = rlst_dynamic_array3!(f64, [3, 5, 2]);
/// ```
#[macro_export]
macro_rules! rlst_dynamic_array3 {
    ($scalar:ty, $shape:expr) => {{
        $crate::dense::array::DynamicArray::<$scalar, 3>::from_shape($shape)
    }};
}

/// Create a new four dimensional heap allocated array.
///
/// A heap allocated array has a size that is determined at runtime.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// // Creates a (3, 5, 2, 4) array with `f64` entries.
/// let arr = rlst_dynamic_array4!(f64, [3, 5, 2, 4]);
/// ```
#[macro_export]
macro_rules! rlst_dynamic_array4 {
    ($scalar:ty, $shape:expr) => {{
        $crate::dense::array::DynamicArray::<$scalar, 4>::from_shape($shape)
    }};
}

/// Create a new five dimensional heap allocated array.
///
/// A heap allocated array has a size that is determined at runtime.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// // Creates a (3, 5, 2, 4, 6) array with `f64` entries.
/// let arr = rlst_dynamic_array5!(f64, [3, 5, 2, 4, 6]);
/// ```
#[macro_export]
macro_rules! rlst_dynamic_array5 {
    ($scalar:ty, $shape:expr) => {{
        $crate::dense::array::DynamicArray::<$scalar, 5>::from_shape($shape)
    }};
}

/// Create a new one dimensional array from a given data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let vec = vec![1.0; 5];
/// let shape = [5];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice1!(f64, vec.as_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice1!(f64, vec.as_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice1 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 1>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};
    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 1>::from_shape($slice, $shape)
    }};
}

/// Create a new two dimensional array from a given data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let vec = vec![1.0; 10];
/// let shape = [2, 5];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice2!(f64, vec.as_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice2!(f64, vec.as_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice2 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 2>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 2>::from_shape($slice, $shape)
    }};
}

/// Create a new three dimensional array from a given data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let vec = vec![1.0; 30];
/// let shape = [2, 5, 3];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice3!(f64, vec.as_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice3!(f64, vec.as_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice3 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 3>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 3>::from_shape($slice, $shape)
    }};
}

/// Create a new four dimensional array from a given data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let vec = vec![1.0; 60];
/// let shape = [2, 5, 3, 2];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice4!(f64, vec.as_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice4!(f64, vec.as_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice4 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 4>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 4>::from_shape($slice, $shape)
    }};
}

/// Create a new five dimensional array from a given data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let vec = vec![1.0; 60];
/// let shape = [2, 5, 3, 2, 1];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice5!(f64, vec.as_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice5!(f64, vec.as_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice5 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 5>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArray::<$scalar, 5>::from_shape($slice, $shape)
    }};
}

/// Create a new one dimensional array from a given mutable data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let mut vec = vec![1.0; 5];
/// let shape = [5];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice_mut1!(f64, vec.as_mut_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice_mut1!(f64, vec.as_mut_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice_mut1 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 1>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 1>::from_shape($slice, $shape)
    }};
}

/// Create a new two dimensional array from a given mutable data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let mut vec = vec![1.0; 10];
/// let shape = [2, 5];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice_mut2!(f64, vec.as_mut_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice_mut2!(f64, vec.as_mut_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice_mut2 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 2>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 2>::from_shape($slice, $shape)
    }};
}

/// Create a new three dimensional array from a given mutable data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let mut vec = vec![1.0; 30];
/// let shape = [2, 5, 3];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice_mut3!(f64, vec.as_mut_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice_mut3!(f64, vec.as_mut_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice_mut3 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 3>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 3>::from_shape($slice, $shape)
    }};
}

/// Create a new four dimensional array from a given mutable data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let mut vec = vec![1.0; 60];
/// let shape = [2, 5, 3, 2];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice_mut4!(f64, vec.as_mut_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice_mut4!(f64, vec.as_mut_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice_mut4 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 4>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 4>::from_shape($slice, $shape)
    }};
}

/// Create a new five dimensional array from a given mutable data slice.
///
/// # Example
/// ```
/// # use rlst::prelude::*;
/// let mut vec = vec![1.0; 60];
/// let shape = [2, 5, 3, 2, 1];
/// let stride = rlst::dense::layout::stride_from_shape(shape);
/// // Specify no stride (use default stride).
/// let arr = rlst_array_from_slice_mut5!(f64, vec.as_mut_slice(), shape);
/// // Specify stride explicitly.
/// let arr = rlst_array_from_slice_mut5!(f64, vec.as_mut_slice(), shape, stride);
/// ```
#[macro_export]
macro_rules! rlst_array_from_slice_mut5 {
    ($scalar:ty, $slice:expr, $shape:expr, $stride:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 5>::from_shape_with_stride(
            $slice, $shape, $stride,
        )
    }};

    ($scalar:ty, $slice:expr, $shape:expr) => {{
        $crate::dense::array::SliceArrayMut::<$scalar, 5>::from_shape($slice, $shape)
    }};
}

// #[cfg(test)]
// mod test {

//     pub use crate::dense::traits::*;

//     #[test]
//     fn create_matrix() {
//         let dim = (2, 3);
//         let mat = rlst_dynamic_mat![f64, dim];

//         assert_eq!(mat.shape(), (2, 3));
//     }

//     // #[test]
//     // fn create_fixed_matrix() {
//     //     let mat = rlst_fixed_mat![f64, 2, 3];

//     //     assert_eq!(mat.shape(), (2, 3));
//     // }

//     #[test]
//     fn create_random_matrix() {
//         let dim = (2, 3);
//         let mat = rlst_rand_mat![f64, dim];

//         assert_eq!(mat.shape(), (2, 3));
//     }

//     #[test]
//     fn create_column_vector() {
//         let length = 5;
//         let vec = rlst_col_vec![f64, length];

//         assert_eq!(vec.shape(), (5, 1));
//     }

//     #[test]
//     fn create_row_vector() {
//         let length = 5;
//         let vec = rlst_row_vec![f64, length];

//         assert_eq!(vec.shape(), (1, 5));
//     }

//     #[test]
//     fn create_random_vector() {
//         let length = 5;
//         let vec = rlst_rand_col_vec![f64, length];

//         assert_eq!(vec.shape(), (5, 1));
//     }

//     #[test]
//     fn create_mut_pointer_mat() {
//         let mut vec = vec![0.0; 10];
//         let ptr = vec.as_mut_ptr();
//         // Create a (2, 5) matrix with stride (1, 2).
//         let mat = unsafe { rlst_mut_pointer_mat!['static, f64, ptr, (2, 5), (1, 2)] };
//         assert_eq!(mat.shape(), (2, 5));
//     }
// }
