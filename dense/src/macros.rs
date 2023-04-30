//! Useful macros

/// Generate a new matrix with C Layout
#[macro_export]
macro_rules! rlst_mat {
    ($ScalarType:ty, $dim:expr) => {{
        use $crate::LayoutType;
        $crate::GenericBaseMatrix::<
            $ScalarType,
            $crate::VectorContainer<$ScalarType>,
            $crate::Dynamic,
            $crate::Dynamic,
        >::from_data(
            $crate::VectorContainer::<$ScalarType>::new($dim.0 * $dim.1),
            $crate::DefaultLayout::from_dimension(($dim.0, $dim.1), (1, $dim.0)),
        )
    }};
}

#[macro_export]
macro_rules! rlst_mut_pointer_mat {
    ($a:lifetime, $ScalarType:ty, $ptr:expr, $dim:expr, $stride:expr) => {{
        let new_layout = $crate::DefaultLayout::new($dim, $stride);
        let nindices = new_layout.convert_2d_raw($dim.0 - 1, $dim.1 - 1) + 1;
        let slice = std::slice::from_raw_parts_mut($ptr, nindices);
        let data = $crate::SliceContainerMut::<$a, Item>::new(slice);

        $crate::SliceMatrixMut::<$a, Item, Dynamic, Dynamic>::from_data(data, new_layout)
    }};
}

#[macro_export]
macro_rules! rlst_pointer_mat {
    ($a:lifetime, $ScalarType:ty, $ptr:expr, $dim:expr, $stride:expr) => {{
        let new_layout = $crate::DefaultLayout::new($dim, $stride);
        let nindices = new_layout.convert_2d_raw($dim.0 - 1, $dim.1 - 1) + 1;
        let slice = std::slice::from_raw_parts($ptr, nindices);
        let data = $crate::SliceContainer::<$a, Item>::new(slice);

        $crate::SliceMatrix::<$a, Item, Dynamic, Dynamic>::from_data(data, new_layout)
    }};
}

#[macro_export]
macro_rules! rlst_rand_mat {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = rand::thread_rng();
        let mut mat = $crate::rlst_mat![$ScalarType, $dim];
        mat.fill_from_standard_normal(&mut rng);
        mat
    }};
}

#[macro_export]
macro_rules! rlst_fixed {
    ($ScalarType:ty, $dim1:literal, $dim2:literal) => {{
        use paste::paste;
        use $crate::LayoutType;
        $crate::GenericBaseMatrix::<
            $ScalarType,
            $crate::ArrayContainer<$ScalarType, { $dim1 * $dim2 }>,
            paste! {[<Fixed $dim1>]},
            paste! {[<Fixed $dim2>]},
        >::from_data(
            $crate::ArrayContainer::<$ScalarType, { $dim1 * $dim2 }>::new(),
            $crate::DefaultLayout::from_dimension(($dim1, $dim2), (1, $dim1)),
        )
    }};
}

#[macro_export]
macro_rules! rlst_col_vec {
    ($ScalarType:ty, $len:expr) => {{
        use $crate::LayoutType;
        $crate::ColumnVectorD::<$ScalarType>::from_data(
            $crate::VectorContainer::<$ScalarType>::new($len),
            $crate::DefaultLayout::from_dimension(($len, 1), (1, $len)),
        )
    }};
}

#[macro_export]
macro_rules! rlst_row_vec {
    ($ScalarType:ty, $len:expr) => {{
        use $crate::LayoutType;
        $crate::RowVectorD::<$ScalarType>::from_data(
            $crate::VectorContainer::<$ScalarType>::new($len),
            $crate::DefaultLayout::from_dimension((1, $len), (1, 1)),
        )
    }};
}

#[macro_export]
macro_rules! rlst_rand_col_vec {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = rand::thread_rng();
        let mut vec = $crate::rlst_col_vec![$ScalarType, $dim];
        vec.fill_from_standard_normal(&mut rng);
        vec
    }};
}

#[macro_export]
macro_rules! rlst_rand_row_vec {
    ($ScalarType:ty, $dim:expr) => {{
        let mut rng = rand::thread_rng();
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
        let mat = rlst_mat![f64, dim];

        assert_eq!(mat.shape(), (2, 3));
    }

    #[test]
    fn create_fixed_matrix() {
        let mat = rlst_fixed![f64, 2, 3];

        assert_eq!(mat.shape(), (2, 3));
    }

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
}
