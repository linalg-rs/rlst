//! A collection of routines to construct matrix objects from scratch or existing data.

use crate::base_matrix::BaseMatrix;
use crate::data_container::ArrayContainer;
use crate::matrix::Matrix;
use crate::matrix_ref::{MatrixRef, MatrixRefMut};
use crate::traits::*;
use crate::types::Scalar;
use crate::{layouts::*, DataContainer, MatrixD};
use rlst_common::traits::constructors::{NewLikeSelf, NewLikeTranspose};
use rlst_common::traits::Identity;
use std::marker::PhantomData;

impl<Item: Scalar, S: SizeIdentifier, MatImpl: MatrixImplTrait<Item, S>> Matrix<Item, MatImpl, S> {
    /// Create a new matrix from a given implementation.
    pub fn new(mat: MatImpl) -> Self {
        Self(mat, PhantomData, PhantomData)
    }

    /// Create a new matrix from a reference to an existing matrix.
    pub fn from_ref(mat: &Matrix<Item, MatImpl, S>) -> crate::RefMat<'_, Item, MatImpl, S> {
        crate::RefMat::new(MatrixRef::new(mat))
    }
}

impl<Item: Scalar, S: SizeIdentifier, MatImpl: MatrixImplTraitMut<Item, S>>
    Matrix<Item, MatImpl, S>
{
    /// Create a new matrix from a reference to an existing matrix.
    pub fn from_ref_mut(
        mat: &mut Matrix<Item, MatImpl, S>,
    ) -> crate::RefMatMut<'_, Item, MatImpl, S> {
        crate::RefMatMut::new(MatrixRefMut::new(mat))
    }
}

impl<Item: Scalar, S: SizeIdentifier, Data: DataContainer<Item = Item>>
    Matrix<Item, BaseMatrix<Item, Data, S>, S>
{
    /// Create a new matrix from a data container and a layout structure.
    pub fn from_data(data: Data, layout: DefaultLayout) -> Self {
        Self::new(BaseMatrix::<Item, Data, S>::new(data, layout))
    }
}

impl<Item: Scalar> MatrixD<Item> {
    /// Create a new matrix from another object that
    /// provides [RandomAccessByValue](rlst_common::traits::RandomAccessByValue) and
    /// [Shape](rlst_common::traits::Shape) traits.
    pub fn from_other<Other: RandomAccessByValue<Item = Item> + Shape>(other: &Other) -> Self {
        let mut mat = crate::rlst_mat![Item, other.shape()];

        for col_index in 0..other.shape().1 {
            for row_index in 0..other.shape().0 {
                mat[[row_index, col_index]] = other.get_value(row_index, col_index).unwrap();
            }
        }

        mat
    }
}

impl<Item: Scalar> Identity for MatrixD<Item> {
    type Out = Self;

    fn identity(shape: (usize, usize)) -> Self::Out {
        let mut ident = crate::rlst_mat![Item, shape];

        for index in 0..std::cmp::min(shape.0, shape.1) {
            ident[[index, index]] = <Item as num::One>::one();
        }

        ident
    }
}

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, S>, S: SizeIdentifier + MatrixBuilder<Item>>
    NewLikeSelf for Matrix<Item, MatImpl, S>
{
    type Out = <S as MatrixBuilder<Item>>::Out;

    fn new_like_self(&self) -> Self::Out {
        <S as MatrixBuilder<Item>>::new_matrix(self.layout().dim())
    }
}

// impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Dynamic>, S: StaticMatrixBuilder<Item>>
//     NewLikeSelf for Matrix<Item, MatImpl, S>
// {
//     type Out = crate::MatrixD<Item>;

//     fn new_like_self(&self) -> Self::Out {
//         crate::rlst_mat![Item, self.layout().dim()]
//     }
// }

impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, Dynamic>> NewLikeTranspose
    for Matrix<Item, MatImpl, Dynamic>
{
    type Out = crate::MatrixD<Item>;

    fn new_like_transpose(&self) -> Self::Out {
        let dim = self.layout().dim();
        crate::rlst_mat![Item, (dim.1, dim.0)]
    }
}

// macro_rules! implement_new_from_self_fixed {
//     ($S:ty) => {
//         impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, $S>> NewLikeSelf
//             for Matrix<Item, MatImpl, RS>
//         {
//             type Out = Matrix<
//                 Item,
//                 BaseMatrix<Item, ArrayContainer<Item, { <$S>::N * <$S>::N }>, $S, $S>,
//                 $S,
//                 $S,
//             >;
//             fn new_like_self(&self) -> Self::Out {
//                 <Self::Out>::from_data(
//                     ArrayContainer::<Item, { <$S>::N * <$S>::N }>::new(),
//                     DefaultLayout::from_dimension((<$S>::N, <$S>::N)),
//                 )
//             }
//         }

//         impl<Item: Scalar, MatImpl: MatrixImplTrait<Item, $RS, $CS>> NewLikeTranspose
//             for Matrix<Item, MatImpl, $RS, $CS>
//         {
//             type Out = Matrix<
//                 Item,
//                 BaseMatrix<Item, ArrayContainer<Item, { <$CS>::N * <$RS>::N }>, $CS, $RS>,
//                 $CS,
//                 $RS,
//             >;
//             fn new_like_transpose(&self) -> Self::Out {
//                 <Self::Out>::from_data(
//                     ArrayContainer::<Item, { <$CS>::N * <$RS>::N }>::new(),
//                     DefaultLayout::from_dimension((<$CS>::N, <$RS>::N)),
//                 )
//             }
//         }
//     };
// }

// implement_new_from_self_fixed!(Fixed2, Fixed2);
// implement_new_from_self_fixed!(Fixed1, Fixed2);

// implement_new_from_self_fixed!(Fixed3, Fixed3);
// implement_new_from_self_fixed!(Fixed1, Fixed3);

// implement_new_from_self_fixed!(Fixed2, Fixed3);
// implement_new_from_self_fixed!(Fixed3, Fixed2);

// implement_new_from_self_fixed!(Fixed2, Fixed1);
// implement_new_from_self_fixed!(Fixed3, Fixed1);
