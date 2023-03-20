//! A collection of routines to construct matrix objects from scratch or existing data.

use crate::base_matrix::BaseMatrix;
use crate::data_container::{ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer};
use crate::layouts::*;
use crate::matrix::{ColumnVectorD, Matrix, RowVectorD, SliceMatrix, SliceMatrixMut};
use crate::traits::*;
use crate::types::{IndexType, Scalar};

// Construct mutable zero matrices

macro_rules! from_zeros_fixed {
    ($RS:ident, $CS:ident, $L:ident) => {
        impl<Item: Scalar>
            Matrix<
                Item,
                BaseMatrix<Item, ArrayContainer<Item, { $RS::N * $CS::N }>, $L, $RS, $CS>,
                $L,
                $RS,
                $CS,
            >
        {
            /// Create a new fixed dimension matrix.
            pub fn zeros_from_dim() -> Self {
                Self::from_data(
                    ArrayContainer::<Item, { $RS::N * $CS::N }>::new(),
                    $L::from_dimension(($RS::N, $CS::N)),
                )
            }
        }
    };
}

macro_rules! from_zeros_fixed_vector {
    ($RS:ident, $CS:ident, $L:ident, $N:expr) => {
        impl<Item: Scalar>
            Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, $N>, $L, $RS, $CS>, $L, $RS, $CS>
        {
            /// Create a new fixed length vector.
            pub fn zeros_from_length() -> Self {
                Self::from_data(ArrayContainer::<Item, $N>::new(), $L::from_length($N))
            }
        }
    };
}

from_zeros_fixed!(Fixed2, Fixed2, RowMajor);
from_zeros_fixed!(Fixed1, Fixed2, RowMajor);

from_zeros_fixed!(Fixed3, Fixed3, RowMajor);
from_zeros_fixed!(Fixed1, Fixed3, RowMajor);

from_zeros_fixed!(Fixed2, Fixed3, RowMajor);
from_zeros_fixed!(Fixed3, Fixed2, RowMajor);

from_zeros_fixed!(Fixed2, Fixed2, ColumnMajor);
from_zeros_fixed!(Fixed1, Fixed2, ColumnMajor);

from_zeros_fixed!(Fixed3, Fixed3, ColumnMajor);
from_zeros_fixed!(Fixed1, Fixed3, ColumnMajor);

from_zeros_fixed!(Fixed2, Fixed3, ColumnMajor);
from_zeros_fixed!(Fixed3, Fixed2, ColumnMajor);

from_zeros_fixed_vector!(Fixed2, Fixed1, ColumnVector, 2);
from_zeros_fixed_vector!(Fixed3, Fixed1, ColumnVector, 3);
from_zeros_fixed_vector!(Fixed1, Fixed2, RowVector, 2);
from_zeros_fixed_vector!(Fixed1, Fixed3, RowVector, 3);

impl<Item: Scalar, L: BaseLayoutType>
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, L, Dynamic, Dynamic>, L, Dynamic, Dynamic>
{
    /// Create a new zero matrix with given number of rows and columns.
    pub fn zeros_from_dim(rows: IndexType, cols: IndexType) -> Self {
        let layout = L::from_dimension((rows, cols));
        Self::from_data(
            VectorContainer::<Item>::new(layout.number_of_elements()),
            L::from_dimension((rows, cols)),
        )
    }
}

impl<Item: Scalar> RowVectorD<Item> {
    /// Create a new zero row vector with given number of elements.
    pub fn zeros_from_length(nelems: IndexType) -> Self {
        Self::from_data(
            VectorContainer::<Item>::new(nelems),
            RowVector::from_length(nelems),
        )
    }
    /// Create a new zero row vector by providing full dimension.
    ///
    /// This routine ensures compatibility to matrix constructors.
    pub fn zeros_from_dim(rows: usize, cols: usize) -> Self {
        assert_eq!(
            rows, 1,
            "Number of rows is {} but must be 1 for row vectors.",
            rows
        );
        Self::from_data(
            VectorContainer::<Item>::new(cols),
            RowVector::from_length(cols),
        )
    }
}

impl<Item: Scalar> ColumnVectorD<Item> {
    /// Create a new zero column vector with given number of elements.
    pub fn zeros_from_length(nelems: IndexType) -> Self {
        Self::from_data(
            VectorContainer::<Item>::new(nelems),
            ColumnVector::from_length(nelems),
        )
    }
    /// Create a new zero column vector by providing full dimension.
    ///
    /// This routine ensures compatibility to matrix constructors.
    pub fn zeros_from_dim(rows: usize, cols: usize) -> Self {
        assert_eq!(
            cols, 1,
            "Number of columns is {} but must be 1 for column vectors.",
            cols
        );
        Self::from_data(
            VectorContainer::<Item>::new(rows),
            ColumnVector::from_length(rows),
        )
    }
}

macro_rules! from_pointer_strided {
    ($RS:ident, $CS:ident, $L:ident) => {
        impl<'a, Item: Scalar> SliceMatrixMut<'a, Item, $L, $RS, $CS> {
            /// Create a new mutable matrix by specifying a pointer, dimension and stride tuple.
            pub unsafe fn from_pointer(
                ptr: *mut Item,
                dim: (IndexType, IndexType),
                stride: (IndexType, IndexType),
            ) -> Self {
                let new_layout = $L::new(dim, stride);
                let nindices = new_layout.convert_2d_raw(dim.0 - 1, dim.1 - 1) + 1;
                let slice = std::slice::from_raw_parts_mut(ptr, nindices);
                let data = SliceContainerMut::<'a, Item>::new(slice);

                SliceMatrixMut::<'a, Item, $L, $RS, $CS>::from_data(data, new_layout)
            }
        }

        impl<'a, Item: Scalar> SliceMatrix<'a, Item, $L, $RS, $CS> {
            /// Create a new matrix by specifying a pointer, dimension and stride tuple.
            pub unsafe fn from_pointer(
                ptr: *const Item,
                dim: (IndexType, IndexType),
                stride: (IndexType, IndexType),
            ) -> Self {
                let new_layout = $L::new(dim, stride);
                let nindices = new_layout.convert_2d_raw(dim.0 - 1, dim.1 - 1) + 1;
                let slice = std::slice::from_raw_parts(ptr, nindices);
                let data = SliceContainer::<'a, Item>::new(slice);

                SliceMatrix::<'a, Item, $L, $RS, $CS>::from_data(data, new_layout)
            }
        }
    };
}

macro_rules! from_pointer {
    ($RS:ident, $CS:ident, $L:ident) => {
        impl<'a, Item: Scalar> SliceMatrixMut<'a, Item, $L, $RS, $CS> {
            /// Create a new mutable matrix from a given pointer and dimension.
            pub unsafe fn from_pointer(ptr: *mut Item, dim: (IndexType, IndexType)) -> Self {
                let new_layout = $L::new(dim);
                let nindices = dim.0 * dim.1;
                let slice = std::slice::from_raw_parts_mut(ptr, nindices);
                let data = SliceContainerMut::<'a, Item>::new(slice);

                SliceMatrixMut::<'a, Item, $L, $RS, $CS>::from_data(data, new_layout)
            }
        }

        impl<'a, Item: Scalar> SliceMatrix<'a, Item, $L, $RS, $CS> {
            /// Create a new matrix from a given pointer and dimension.
            pub unsafe fn from_pointer(ptr: *const Item, dim: (IndexType, IndexType)) -> Self {
                let new_layout = $L::new(dim);
                let nindices = dim.0 * dim.1;
                let slice = std::slice::from_raw_parts(ptr, nindices);
                let data = SliceContainer::<'a, Item>::new(slice);

                SliceMatrix::<'a, Item, $L, $RS, $CS>::from_data(data, new_layout)
            }
        }
    };
}

from_pointer!(Dynamic, Dynamic, RowMajor);
from_pointer!(Dynamic, Dynamic, ColumnMajor);
from_pointer_strided!(Dynamic, Dynamic, ArbitraryStrideColumnMajor);
from_pointer_strided!(Dynamic, Dynamic, ArbitraryStrideRowMajor);
