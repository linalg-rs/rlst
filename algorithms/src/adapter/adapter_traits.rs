//! Adapter traits to transform a custom data type into a type that ```rlst-algorithms``` understands.
use crate::adapter::dense_matrix::DenseContainer;

/// This trait provides a method ```algorithms``` that converts a type into a type implementing the ```DenseMatrixInterface```.
pub trait AlgorithmsAdapter {
    type T;
    type OutputAdapter<'a>: super::dense_matrix::DenseContainerInterface<T = Self::T>
    where
        Self: 'a;

    fn algorithms<'a>(&'a self) -> DenseContainer<Self::OutputAdapter<'a>>;
}

/// This trait provides a method ```algorithms``` that converts a type into a type implementing the ```DenseMatrixInterfaceMut```.
pub trait AlgorithmsAdapterMut {
    type T;
    type OutputAdapter<'a>: super::dense_matrix::DenseContainerInterfaceMut<T = Self::T>
    where
        Self: 'a;

    fn algorithms_mut<'a>(&'a mut self) -> DenseContainer<Self::OutputAdapter<'a>>;
}
