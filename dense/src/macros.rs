//! The macros defined here make it easy to create new matrices and vectors.

pub use rand;

/// Create a new matrix.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// // Creates a (3, 5) matrix with `f64` entries.
/// let mat = rlst_dynamic_mat![f64, (3, 5)];
/// ```
#[macro_export]
macro_rules! rlst_dynamic_mat {
    ($ScalarType:ty, $dim:expr) => {{
        use $crate::{Dynamic, MatrixBuilder};
        <Dynamic as MatrixBuilder<$ScalarType>>::new_matrix($dim)
    }};
}
#[macro_export]
macro_rules! rlst_static_mat {
    ($ScalarType:ty, $SizeIdentifier:ty) => {{
        use $crate::{MatrixBuilder};
        use $crate::SizeIdentifierValue;
        match <$SizeIdentifier>::SIZE {
            SizeIdentifierValue::Static(m, n) => <$SizeIdentifier as MatrixBuilder<$ScalarType>>::new_matrix((m, n)),
            _ => panic!("Macro `rlst_static_mat` can only be called with size identifiers representic static sizes."),
        }
    }};
}

/// Create a new matrix from a given mutable pointer.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// let mut vec = vec![0.0; 10];
/// vec[5] = 5.0;
/// let ptr = vec.as_mut_ptr();
/// // Create a (2, 5) matrix with stride (1, 2).
/// let mat = unsafe {rlst_mut_pointer_mat!['static, f64, ptr, (2, 5), (1, 2)] };
/// assert_eq!(mat[[1, 2]], 5.0);
/// ```
#[macro_export]
macro_rules! rlst_mut_pointer_mat {
    ($a:lifetime, $ScalarType:ty, $ptr:expr, $dim:expr, $stride:expr) => {{
        let new_layout = $crate::DefaultLayout::new($dim, $stride);
        let nindices = new_layout.convert_2d_raw($dim.0 - 1, $dim.1 - 1) + 1;
        let slice = std::slice::from_raw_parts_mut($ptr, nindices);
        let data = $crate::SliceContainerMut::<$a, $ScalarType>::new(slice);

        $crate::SliceMatrixMut::<$a, $ScalarType, Dynamic>::from_data(data, new_layout)
    }};
}

/// Create a new matrix from a given pointer.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::traits::*;
/// let vec = vec![0.0; 10];
/// let ptr = vec.as_ptr();
/// // Create a (2, 5) matrix with stride (1, 2).
/// let mat = unsafe {rlst_pointer_mat!['static, f64, ptr, (2, 5), (1, 2)] };
/// # assert_eq!(mat.shape(), (2, 5));
/// # assert_eq!(mat.layout().stride(), (1, 2));
/// ```
#[macro_export]
macro_rules! rlst_pointer_mat {
    ($a:lifetime, $ScalarType:ty, $ptr:expr, $dim:expr, $stride:expr) => {{
        let new_layout = $crate::DefaultLayout::new($dim, $stride);
        let nindices = new_layout.convert_2d_raw($dim.0 - 1, $dim.1 - 1) + 1;
        let slice = std::slice::from_raw_parts($ptr, nindices);
        let data = $crate::SliceContainer::<$a, $ScalarType>::new(slice);

        $crate::SliceMatrix::<$a, $ScalarType, Dynamic>::from_data(data, new_layout)
    }};
}

/// Create a new random matrix with normally distributed elements.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::types::*;
/// // Creates a (3, 5) random matrix with `c64` entries.
/// let mat = rlst_rand_mat![c64, (3, 5)];
/// ```
#[macro_export]
macro_rules! rlst_rand_mat {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = $crate::macros::rand::thread_rng();
        let mut mat = $crate::rlst_dynamic_mat![$ScalarType, $dim];
        mat.fill_from_standard_normal(&mut rng);
        mat
    }};
}

// / Create a new matrix with compile-time known dimension parameters and stack allocated storage.
// /
// / Currently only dimensions 1, 2, 3 are supported as
// / dimension parameters.
// /
// / Example:
// / ```
// / # use rlst_dense::*;
// / # use rlst_common::traits::*;
// / // Creates a (2, 3) .
// / let mat = rlst_fixed_mat![f64, 2, 3];
// / # assert_eq!(mat.shape(), (2, 3))
// / ```
// #[macro_export]
// macro_rules! rlst_fixed_mat {
//     ($ScalarType:ty, $dim1:literal, $dim2:literal) => {{
//         use paste::paste;
//         use $crate::LayoutType;
//         #[allow(unused_imports)]
//         use $crate::{Fixed2, Fixed3};
//         $crate::GenericBaseMatrix::<
//             $ScalarType,
//             $crate::ArrayContainer<$ScalarType, { $dim1 * $dim2 }>,
//             paste! {[<Fixed $dim1>]},
//             paste! {[<Fixed $dim2>]},
//         >::from_data(
//             $crate::ArrayContainer::<$ScalarType, { $dim1 * $dim2 }>::new(),
//             $crate::DefaultLayout::from_dimension(($dim1, $dim2)),
//         )
//     }};
// }

// /// Create a new random matrix with compile-time known dimension parameters and stack allocated storage.
// ///
// /// Currently only dimensions 1, 2, 3 are supported as
// /// dimension parameters.
// ///
// /// Example:
// /// ```
// /// # use rlst_dense::*;
// /// # use rlst_common::traits::*;
// /// // Creates a (2, 3) .
// /// let mat = rlst_static_rand_mat![f64, 2, 3];
// /// # assert_eq!(mat.shape(), (2, 3))
// /// ```
// #[macro_export]
// macro_rules! rlst_static_rand_mat {
//     ($ScalarType:ty, $dim1:literal, $dim2:literal) => {{
//         let mut rng = $crate::macros::rand::thread_rng();
//         let mut mat = $crate::rlst_fixed_mat![$ScalarType, $dim1, $dim2];
//         mat.fill_from_standard_normal(&mut rng);
//         mat
//     }};
// }

/// Create a new column vector.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::traits::*;
/// // Creates a column vector with 5 elements.
/// let vec = rlst_col_vec![f64, 5];
/// # assert_eq!(vec.shape(), (5, 1));
/// # assert_eq!(vec.layout().stride(), (1, 5));
/// ```
#[macro_export]
macro_rules! rlst_col_vec {
    ($ScalarType:ty, $len:expr) => {{
        $crate::rlst_dynamic_mat![$ScalarType, ($len, 1)]
    }};
}

/// Create a new row vector.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::traits::*;
/// // Creates a row vector with 5 elements.
/// let vec = rlst_row_vec![f64, 5];
/// # assert_eq!(vec.shape(), (1, 5));
/// # assert_eq!(vec.layout().stride(), (1, 1));
/// ```
#[macro_export]
macro_rules! rlst_row_vec {
    ($ScalarType:ty, $len:expr) => {{
        $crate::rlst_dynamic_mat![$ScalarType, (1, $len)]
    }};
}

/// Create a new random column vector with normally distributed elements.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::traits::*;
/// // Creates a random column vector with 5 elements.
/// let vec = rlst_rand_col_vec![f64, 5];
/// # assert_eq!(vec.shape(), (5, 1));
/// ```
#[macro_export]
macro_rules! rlst_rand_col_vec {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = $crate::macros::rand::thread_rng();
        let mut vec = $crate::rlst_col_vec![$ScalarType, $dim];
        vec.fill_from_standard_normal(&mut rng);
        vec
    }};
}

/// Create a new random row vector with normally distributed elements.
///
/// Example:
/// ```
/// # use rlst_dense::*;
/// # use rlst_common::traits::*;
/// // Creates a random column vector with 5 elements.
/// let vec = rlst_rand_row_vec![f64, 5];
/// # assert_eq!(vec.shape(), (1, 5));
/// ```
#[macro_export]
macro_rules! rlst_rand_row_vec {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = $crate::macros::rand::thread_rng();
        let mut vec = $crate::rlst_row_vec![$ScalarType, $dim];
        vec.fill_from_standard_normal(&mut rng);
        vec
    }};
}

#[cfg(test)]
mod test {

    pub use crate::traits::*;

    #[test]
    fn create_matrix() {
        let dim = (2, 3);
        let mat = rlst_dynamic_mat![f64, dim];

        assert_eq!(mat.shape(), (2, 3));
    }

    // #[test]
    // fn create_fixed_matrix() {
    //     let mat = rlst_fixed_mat![f64, 2, 3];

    //     assert_eq!(mat.shape(), (2, 3));
    // }

    #[test]
    fn create_random_matrix() {
        let dim = (2, 3);
        let mat = rlst_rand_mat![f64, dim];

        assert_eq!(mat.shape(), (2, 3));
    }

    #[test]
    fn create_column_vector() {
        let length = 5;
        let vec = rlst_col_vec![f64, length];

        assert_eq!(vec.shape(), (5, 1));
    }

    #[test]
    fn create_row_vector() {
        let length = 5;
        let vec = rlst_row_vec![f64, length];

        assert_eq!(vec.shape(), (1, 5));
    }

    #[test]
    fn create_random_vector() {
        let length = 5;
        let vec = rlst_rand_col_vec![f64, length];

        assert_eq!(vec.shape(), (5, 1));
    }

    #[test]
    fn create_mut_pointer_mat() {
        let mut vec = vec![0.0; 10];
        let ptr = vec.as_mut_ptr();
        // Create a (2, 5) matrix with stride (1, 2).
        let mat = unsafe { rlst_mut_pointer_mat!['static, f64, ptr, (2, 5), (1, 2)] };
        assert_eq!(mat.shape(), (2, 5));
    }
}
