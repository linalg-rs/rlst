//! Useful macros

/// Generate a new matrix with C Layout
#[macro_export]
macro_rules! mat {
    ($ScalarType:ty, $dim:expr) => {
        mat![$ScalarType, $dim, RowMajor]
    };
    ($ScalarType:ty, $dim:expr, RowMajor) => {
        $crate::GenericBaseMatrixMut::<
            $ScalarType,
            $crate::RowMajor,
            $crate::VectorContainer<$ScalarType>,
            $crate::Dynamic,
            $crate::Dynamic,
        >::zeros_from_dim($dim.0, $dim.1)
    };
    ($ScalarType:ty, $dim:expr, ColumnMajor) => {
        $crate::GenericBaseMatrixMut::<
            $ScalarType,
            $crate::ColumnMajor,
            $crate::VectorContainer<$ScalarType>,
            $crate::Dynamic,
            $crate::Dynamic,
        >::zeros_from_dim($dim.0, $dim.1)
    };
}

#[macro_export]
macro_rules! rand_mat {
    ($ScalarType:ty, $dim:expr) => {
        rand_mat![$ScalarType, $dim, RowMajor]
    };
    ($ScalarType:ty, $dim:expr, $layout:tt) => {{
        let mut rng = rand::thread_rng();
        let mut mat = $crate::mat![$ScalarType, $dim, $layout];
        mat.fill_from_rand_standard_normal(&mut rng);
        mat
    }};
}

#[macro_export]
macro_rules! vector {
    ($ScalarType:ty, $len:expr) => {
        vector![$ScalarType, $len, ColumnVector]
    };
    ($ScalarType:ty, $len:expr, ColumnVector) => {
        $crate::ColumnVectorD::<$ScalarType>::zeros_from_length($len)
    };
    ($ScalarType:ty, $len:expr, RowVector) => {
        $crate::RowVectorD::<$ScalarType>::zeros_from_length($len)
    };
}

#[macro_export]
macro_rules! rand_vector {
    ($ScalarType:ty, $dim:expr) => {
        rand_vector![$ScalarType, $dim, ColumnVector]
    };
    ($ScalarType:ty, $dim:expr, $orientation:tt) => {{
        let mut rng = rand::thread_rng();
        let mut vec = $crate::vector![$ScalarType, $dim, $orientation];
        vec.fill_from_rand_standard_normal(&mut rng);
        vec
    }};
}

#[cfg(test)]
mod test {

    #[test]
    fn create_row_major_matrix() {
        let dim = (2, 3);
        let mat = mat![f64, dim];

        assert_eq!(mat.dim(), (2, 3));
    }

    #[test]
    fn create_random_matrix() {
        let dim = (2, 3);
        let mat = rand_mat![f64, dim];

        assert_eq!(mat.dim(), (2, 3));
    }

    #[test]
    fn create_column_vector() {
        let length = 5;
        let vec = vector![f64, length];

        assert_eq!(vec.dim(), (5, 1));
    }

    #[test]
    fn create_row_vector() {
        let length = 5;
        let vec = vector![f64, length, RowVector];

        assert_eq!(vec.dim(), (1, 5));
    }

    #[test]
    fn create_random_vector() {
        let length = 5;
        let vec = rand_vector![f64, length];

        assert_eq!(vec.dim(), (5, 1));
    }
}
