//! Adapter traits to transform a custom data type into a type that ```rlst-algorithms``` understands.
use crate::adapter::dense_matrix::DenseMatrix;

/// This trait provides a method ```algorithms``` that converts a type into a type implementing the ```DenseMatrixInterface```.
pub trait AlgorithmsAdapter {
    type T;
    type OutputAdapter<'a>: super::dense_matrix::DenseMatrixInterface<T = Self::T>
    where
        Self: 'a;

    fn algorithms<'a>(&'a self) -> DenseMatrix<Self::OutputAdapter<'a>>;
}

/// This trait provides a method ```algorithms``` that converts a type into a type implementing the ```DenseMatrixInterfaceMut```.
pub trait AlgorithmsAdapterMut {
    type T;
    type OutputAdapter<'a>: super::dense_matrix::DenseMatrixInterfaceMut<T = Self::T>
    where
        Self: 'a;

    fn algorithms_mut<'a>(&'a mut self) -> DenseMatrix<Self::OutputAdapter<'a>>;
}
