//! A collection of routines to construct matrix objects from scratch or existing data.

use crate::base_matrix::BaseMatrix;
use crate::data_container::{ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer};
use crate::layouts::*;
use crate::matrix::{Matrix, SliceMatrix, SliceMatrixMut};
use crate::traits::*;
use crate::types::Scalar;

// Construct mutable zero matrices

macro_rules! from_zeros_fixed {
    ($RS:ident, $CS:ident) => {
        impl<Item: Scalar>
            Matrix<
                Item,
                BaseMatrix<Item, ArrayContainer<Item, { $RS::N * $CS::N }>, $RS, $CS>,
                $RS,
                $CS,
            >
        {
            /// Create a new fixed dimension matrix.
            pub fn zeros_from_dim() -> Self {
                Self::from_data(
                    ArrayContainer::<Item, { $RS::N * $CS::N }>::new(),
                    DefaultLayout::from_dimension(($RS::N, $CS::N), (1, $RS::N)),
                )
            }
        }
    };
}

from_zeros_fixed!(Fixed2, Fixed2);
from_zeros_fixed!(Fixed1, Fixed2);

from_zeros_fixed!(Fixed3, Fixed3);
from_zeros_fixed!(Fixed1, Fixed3);

from_zeros_fixed!(Fixed2, Fixed3);
from_zeros_fixed!(Fixed3, Fixed2);

from_zeros_fixed!(Fixed2, Fixed1);
from_zeros_fixed!(Fixed3, Fixed1);

impl<Item: Scalar>
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, Dynamic, Dynamic>, Dynamic, Dynamic>
{
    /// Create a new zero matrix with given number of rows and columns.
    pub fn zeros_from_dim(rows: usize, cols: usize) -> Self {
        let layout = DefaultLayout::from_dimension((rows, cols), (1, rows));
        Self::from_data(
            VectorContainer::<Item>::new(layout.number_of_elements()),
            layout,
        )
    }
}

impl<Item: Scalar>
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, Dynamic, Fixed1>, Dynamic, Fixed1>
{
    /// Create a new zero column vector with a given number of entries.
    pub fn zeros_from_length(nelems: usize) -> Self {
        let layout = DefaultLayout::from_dimension((nelems, 1), (1, nelems));
        Self::from_data(
            VectorContainer::<Item>::new(layout.number_of_elements()),
            layout,
        )
    }
}

impl<Item: Scalar>
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, Fixed1, Dynamic>, Fixed1, Dynamic>
{
    /// Create a new zero row vector with a given number of entries.
    pub fn zeros_from_length(nelems: usize) -> Self {
        let layout = DefaultLayout::from_dimension((1, nelems), (1, 1));
        Self::from_data(
            VectorContainer::<Item>::new(layout.number_of_elements()),
            layout,
        )
    }
}

macro_rules! from_pointer_strided {
    ($RS:ident, $CS:ident) => {
        impl<'a, Item: Scalar> SliceMatrixMut<'a, Item, $RS, $CS> {
            /// Create a new mutable matrix by specifying a pointer, dimension and stride tuple.
            pub unsafe fn from_pointer(
                ptr: *mut Item,
                dim: (usize, usize),
                stride: (usize, usize),
            ) -> Self {
                let new_layout = DefaultLayout::new(dim, stride);
                let nindices = new_layout.convert_2d_raw(dim.0 - 1, dim.1 - 1) + 1;
                let slice = std::slice::from_raw_parts_mut(ptr, nindices);
                let data = SliceContainerMut::<'a, Item>::new(slice);

                SliceMatrixMut::<'a, Item, $RS, $CS>::from_data(data, new_layout)
            }
        }

        impl<'a, Item: Scalar> SliceMatrix<'a, Item, $RS, $CS> {
            /// Create a new matrix by specifying a pointer, dimension and stride tuple.
            pub unsafe fn from_pointer(
                ptr: *const Item,
                dim: (usize, usize),
                stride: (usize, usize),
            ) -> Self {
                let new_layout = DefaultLayout::new(dim, stride);
                let nindices = new_layout.convert_2d_raw(dim.0 - 1, dim.1 - 1) + 1;
                let slice = std::slice::from_raw_parts(ptr, nindices);
                let data = SliceContainer::<'a, Item>::new(slice);

                SliceMatrix::<'a, Item, $RS, $CS>::from_data(data, new_layout)
            }
        }
    };
}

from_pointer_strided!(Dynamic, Dynamic);
