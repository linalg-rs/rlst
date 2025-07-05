//! A collection of macros for Rlst.

/// Dot product between compatible arrays.
///
/// It always allocates a new array for the result.
#[macro_export]
macro_rules! dot {
    ($a:expr, $b:expr) => {{
        $crate::traits::linalg::base::MultIntoResize::simple_mult_into_resize(
            $crate::dense::array::empty_array(),
            $a,
            $b,
        )
    }};

    ($a:expr, $b:expr, $($c:expr),+) => {{
        dot!($a, dot!($b, $($c),+))
    }};

}

/// Create a new n-dimensional diagonal array from a list of diagonal entries.
///
/// Call with `diag!(d)` where `d` is a one-dimensional array to obtain a matrix with `d` on the
/// diagonal. For n-dimensional diagonal arrays, call with `diag!(d, NDIM)` where `NDIM` is the
/// number of dimensions and `d` is a list of diagonal entries.
#[macro_export]
macro_rules! diag {
    ($d:expr) => {
        diag!($d, 2)
    };

    ($d:expr, $ndim:literal) => {{
        use itertools::izip;
        use $crate::traits::{array::Len, iterators::ArrayIterator, iterators::GetDiagMut};

        let mut diag_array =
            $crate::dense::array::DynArray::<_, $ndim>::from_shape([$d.len(); $ndim]);
        izip!(diag_array.diag_iter_mut(), $d.iter()).for_each(|(diag_entry, entry)| {
            *diag_entry = entry;
        });
        diag_array
    }};
}

// /// Create a new rank1 array of the form `u x v^T`.
// ///
// #[macro_export]
// macro_rules! rlst_rank1_array {
//     ($u:expr, $v:expr) => {
//         $crate::dense::array::rank1_array::Rank1Array::new_array($u, $v)
//     };
// }
//
// /// Create a new two one dimensional heap allocated array.
// ///
// /// A heap allocated array has a size that is determined at runtime.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// // Creates a (3) array with `f64` entries.
// /// let arr = rlst_dynamic_array1!(f64, [3]);
// /// ```
// #[macro_export]
// macro_rules! rlst_dynamic_array1 {
//     ($scalar:ty, $shape:expr) => {{
//         $crate::dense::array::DynamicArray::<$scalar, 1>::from_shape($shape)
//     }};
// }
//
// /// Create a new two dimensional heap allocated array.
// ///
// /// A heap allocated array has a size that is determined at runtime.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// // Creates a (3, 5) array with `f64` entries.
// /// let arr = rlst_dynamic_array2!(f64, [3, 5]);
// /// ```
// #[macro_export]
// macro_rules! rlst_dynamic_array2 {
//     ($scalar:ty, $shape:expr) => {{
//         $crate::dense::array::DynamicArray::<$scalar, 2>::from_shape($shape)
//     }};
// }
//
// /// Create a new three dimensional heap allocated array.
// ///
// /// A heap allocated array has a size that is determined at runtime.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// // Creates a (3, 5, 2) array with `f64` entries.
// /// let arr = rlst_dynamic_array3!(f64, [3, 5, 2]);
// /// ```
// #[macro_export]
// macro_rules! rlst_dynamic_array3 {
//     ($scalar:ty, $shape:expr) => {{
//         $crate::dense::array::DynamicArray::<$scalar, 3>::from_shape($shape)
//     }};
// }
//
// /// Create a new four dimensional heap allocated array.
// ///
// /// A heap allocated array has a size that is determined at runtime.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// // Creates a (3, 5, 2, 4) array with `f64` entries.
// /// let arr = rlst_dynamic_array4!(f64, [3, 5, 2, 4]);
// /// ```
// #[macro_export]
// macro_rules! rlst_dynamic_array4 {
//     ($scalar:ty, $shape:expr) => {{
//         $crate::dense::array::DynamicArray::<$scalar, 4>::from_shape($shape)
//     }};
// }
//
// /// Create a new five dimensional heap allocated array.
// ///
// /// A heap allocated array has a size that is determined at runtime.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// // Creates a (3, 5, 2, 4, 6) array with `f64` entries.
// /// let arr = rlst_dynamic_array5!(f64, [3, 5, 2, 4, 6]);
// /// ```
// #[macro_export]
// macro_rules! rlst_dynamic_array5 {
//     ($scalar:ty, $shape:expr) => {{
//         $crate::dense::array::DynamicArray::<$scalar, 5>::from_shape($shape)
//     }};
// }
//
// /// Create a new one dimensional array from a given data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let vec = vec![1.0; 5];
// /// let shape = [5];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice1!(vec.as_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice1!(vec.as_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice1 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArray::<_, 1>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArray::<_, 1>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new two dimensional array from a given data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let vec = vec![1.0; 10];
// /// let shape = [2, 5];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice2!(vec.as_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice2!(vec.as_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice2 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArray::<_, 2>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArray::<_, 2>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new three dimensional array from a given data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let vec = vec![1.0; 30];
// /// let shape = [2, 5, 3];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice3!(vec.as_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice3!(vec.as_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice3 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArray::<_, 3>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArray::<_, 3>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new four dimensional array from a given data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let vec = vec![1.0; 60];
// /// let shape = [2, 5, 3, 2];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice4!(vec.as_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice4!(vec.as_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice4 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArray::<_, 4>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArray::<_, 4>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new five dimensional array from a given data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let vec = vec![1.0; 60];
// /// let shape = [2, 5, 3, 2, 1];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice5!(vec.as_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice5!(vec.as_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice5 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArray::<_, 5>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArray::<_, 5>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new one dimensional array from a given mutable data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let mut vec = vec![1.0; 5];
// /// let shape = [5];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice_mut1!(vec.as_mut_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice_mut1!(vec.as_mut_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice_mut1 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArrayMut::<_, 1>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArrayMut::<_, 1>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new two dimensional array from a given mutable data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let mut vec = vec![1.0; 10];
// /// let shape = [2, 5];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice_mut2!(vec.as_mut_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice_mut2!(vec.as_mut_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice_mut2 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArrayMut::<_, 2>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArrayMut::<_, 2>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new three dimensional array from a given mutable data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let mut vec = vec![1.0; 30];
// /// let shape = [2, 5, 3];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice_mut3!(vec.as_mut_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice_mut3!(vec.as_mut_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice_mut3 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArrayMut::<_, 3>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArrayMut::<_, 3>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new four dimensional array from a given mutable data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let mut vec = vec![1.0; 60];
// /// let shape = [2, 5, 3, 2];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice_mut4!(vec.as_mut_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice_mut4!(vec.as_mut_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice_mut4 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArrayMut::<_, 4>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArrayMut::<_, 4>::from_shape($slice, $shape)
//     }};
// }
//
// /// Create a new five dimensional array from a given mutable data slice.
// ///
// /// # Example
// /// ```
// /// # use rlst::prelude::*;
// /// let mut vec = vec![1.0; 60];
// /// let shape = [2, 5, 3, 2, 1];
// /// let stride = rlst::dense::layout::stride_from_shape(shape);
// /// // Specify no stride (use default stride).
// /// let arr = rlst_array_from_slice_mut5!(vec.as_mut_slice(), shape);
// /// // Specify stride explicitly.
// /// let arr = rlst_array_from_slice_mut5!(vec.as_mut_slice(), shape, stride);
// /// ```
// #[macro_export]
// macro_rules! rlst_array_from_slice_mut5 {
//     ($slice:expr, $shape:expr, $stride:expr) => {{
//         $crate::dense::array::StridedSliceArrayMut::<_, 5>::from_shape_and_stride(
//             $slice, $shape, $stride,
//         )
//     }};
//
//     ($slice:expr, $shape:expr) => {{
//         $crate::dense::array::SliceArrayMut::<_, 5>::from_shape($slice, $shape)
//     }};
// }
//
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
