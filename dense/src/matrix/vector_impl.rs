//! Specialisation for vectors
//!

use crate::base_matrix::BaseMatrix;
use crate::matrix::Matrix;
use crate::data_container::{ArrayContainer, VectorContainer, DataContainer};
use crate::traits::*;
use crate::layouts::*;
use crate::types::{IndexType, HScalar};

macro_rules! vec_dynamic {
    ($MatrixType:ident, $BaseType:ident, $Layout:ident, $RS:ident, $CS:ident) => {
        impl<Item: HScalar>
            $MatrixType<
                Item,
                $BaseType<Item, VectorContainer<Item>, $Layout, $RS, $CS>,
                $Layout,
                $RS,
                $CS,
            >
        {
            pub fn from_data(data: VectorContainer<Item>) -> Self {
                Self::new($BaseType::<
                    Item,
                    VectorContainer<Item>,
                    $Layout,
                    $RS,
                    $CS,
                >::new(data))
            }

            pub fn from_zeros(length: IndexType) -> Self {
                Self::from_data(VectorContainer::<Item>::new(length))
            }
        }
    };
}

macro_rules! vec_fixed {
    ($MatrixType:ident, $BaseType:ident, $Layout:ident, $RS:ident, $CS:ident, $N:expr) => {
        impl<Item: HScalar>
            $MatrixType<
                Item,
                $BaseType<Item, ArrayContainer<Item, $N>, $Layout, $RS, $CS>,
                $Layout,
                $RS,
                $CS,
            >
        {
            pub fn from_data(data: ArrayContainer<Item, $N>) -> Self {
                Self::new($BaseType::<
                    Item,
                    ArrayContainer<Item, $N>,
                    $Layout,
                    $RS,
                    $CS,
                >::new(data))
            }

            pub fn from_zeros() -> Self {
                Self::from_data(ArrayContainer::<Item, $N>::new())
            }
        }
    };
}

vec_dynamic!(Matrix, BaseMatrix, RowMajor, Dynamic, Fixed1);
vec_dynamic!(Matrix, BaseMatrix, RowMajor, Fixed1, Dynamic);


vec_fixed!(Matrix, BaseMatrix, RowMajor, Fixed2, Fixed1, 2);
vec_fixed!(Matrix, BaseMatrix, RowMajor, Fixed3, Fixed1, 2);
vec_fixed!(Matrix, BaseMatrix, RowMajor, Fixed1, Fixed2, 2);
vec_fixed!(Matrix, BaseMatrix, RowMajor, Fixed1, Fixed3, 2);


impl<Item: HScalar, Data: DataContainer<Item=Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    Matrix<Item, BaseMatrix<Item, Data, VLayout, RS, CS>, VLayout, RS, CS> {
        pub fn length(&self) -> IndexType {
            self.0.length()
        }
    }

