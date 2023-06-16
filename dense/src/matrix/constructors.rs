//! A collection of routines to construct matrix objects from scratch or existing data.

use crate::base_matrix::BaseMatrix;
use crate::data_container::ArrayContainer;
use crate::matrix::Matrix;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::Scalar;
use crate::{layouts::*, DataContainer};
use rlst_common::traits::constructors::NewLikeSelf;
use std::marker::PhantomData;

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        MatImpl: MatrixImplTrait<Item, RS, CS>,
    > Matrix<Item, MatImpl, RS, CS>
{
    /// Create a new matrix from a given implementation.
    pub fn new(mat: MatImpl) -> Self {
        Self(mat, PhantomData, PhantomData, PhantomData)
    }

    /// Create a new matrix from a reference to an existing matrix.
    pub fn from_ref(
        mat: &Matrix<Item, MatImpl, RS, CS>,
    ) -> crate::RefMat<'_, Item, MatImpl, RS, CS> {
        crate::RefMat::new(MatrixRef::new(mat))
    }
}

impl<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainer<Item = Item>>
    Matrix<Item, BaseMatrix<Item, Data, RS, CS>, RS, CS>
{
    /// Create a new matrix from a data container and a layout structure.
    pub fn from_data(data: Data, layout: DefaultLayout) -> Self {
        Self::new(BaseMatrix::<Item, Data, RS, CS>::new(data, layout))
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Dynamic, Dynamic>> NewLikeSelf
    for Matrix<Item, MatImpl, Dynamic, Dynamic>
{
    type Out = crate::MatrixD<Item>;

    fn new_like_self(&self) -> Self::Out {
        crate::rlst_mat![Item, self.layout().dim()]
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Fixed1, Dynamic>> NewLikeSelf
    for Matrix<Item, MatImpl, Fixed1, Dynamic>
{
    type Out = crate::RowVectorD<Item>;

    fn new_like_self(&self) -> Self::Out {
        crate::rlst_row_vec![Item, self.layout().number_of_elements()]
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Dynamic, Fixed1>> NewLikeSelf
    for Matrix<Item, MatImpl, Dynamic, Fixed1>
{
    type Out = crate::ColumnVectorD<Item>;

    fn new_like_self(&self) -> Self::Out {
        crate::rlst_col_vec![Item, self.layout().number_of_elements()]
    }
}

macro_rules! implement_new_from_self_fixed {
    ($RS:ty, $CS:ty) => {
        impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, $RS, $CS>> NewLikeSelf
            for Matrix<Item, MatImpl, $RS, $CS>
        {
            type Out = Matrix<
                Item,
                BaseMatrix<Item, ArrayContainer<Item, { <$RS>::N * <$CS>::N }>, $RS, $CS>,
                $RS,
                $CS,
            >;
            fn new_like_self(&self) -> Self::Out {
                <Self::Out>::from_data(
                    ArrayContainer::<Item, { <$RS>::N * <$CS>::N }>::new(),
                    DefaultLayout::from_dimension((<$RS>::N, <$CS>::N), (1, <$RS>::N)),
                )
            }
        }
    };
}

implement_new_from_self_fixed!(Fixed2, Fixed2);
implement_new_from_self_fixed!(Fixed1, Fixed2);

implement_new_from_self_fixed!(Fixed3, Fixed3);
implement_new_from_self_fixed!(Fixed1, Fixed3);

implement_new_from_self_fixed!(Fixed2, Fixed3);
implement_new_from_self_fixed!(Fixed3, Fixed2);

implement_new_from_self_fixed!(Fixed2, Fixed1);
implement_new_from_self_fixed!(Fixed3, Fixed1);
