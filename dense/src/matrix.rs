//! Definition of the main matrix type.
//!
//! [Matrix] is a basic type that provides an interface to dealing
//! with generic matrices. The only member of this type is an implementation
//! to which all calls are forwarded. Implementations can represent more basic
//! types, addition operations on matrices, scalar multiplications, or other
//! types of implementations. The only condition is that the implementation itself
//! implements [MatrixImplTrait] or [MatrixImplTraitMut].
//! A matrix is generic over the following parameters:
//! - `Item`. Implements the [Scalar] trait and represents the underlying scalar type
//!           of the matrix.
//! - `MatImpl`. The actual implementation of the matrix. It must itself implement the
//!              trait [MatrixImplTrait] or [MatrixImplTraitMut] depending on whether mutable access
//!              is required.
//! - `S`. A type that implements [SizeIdentifier] and specifies whether the dimensions are not at
//!        compile time or dynamically at runtime.

pub mod common_impl;
pub mod constructors;
pub mod iterators;
pub mod matrix_slices;
pub mod operations;
pub mod random;

use crate::base_matrix::BaseMatrix;
use crate::data_container::{ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer};
use crate::matrix_ref::{MatrixRef, MatrixRefMut};
use crate::matrix_view::{MatrixView, MatrixViewMut};
use crate::traits::*;
use crate::types::Scalar;
use std::marker::PhantomData;

/// A [RefMat] is a matrix whose implementation is a reference to another matrix.
/// This is used to convert a reference to a matrix to an owned matrix whose implementation
/// is a reference to matrix.
pub type RefMat<'a, Item, MatImpl, S> = Matrix<Item, MatrixRef<'a, Item, MatImpl, S>, S>;

/// A [RefMatMut] is a matrix whose implementation is a reference to another matrix.
/// This is used to convert a reference to a matrix to an owned matrix whose implementation
/// is a reference to matrix.
pub type RefMatMut<'a, Item, MatImpl, S> = Matrix<Item, MatrixRefMut<'a, Item, MatImpl, S>, S>;

/// This type represents a generic matrix that depends on a [BaseMatrix] type.
pub type GenericBaseMatrix<Item, Data, S> = Matrix<Item, BaseMatrix<Item, Data, S>, S>;

/// A [SliceMatrix] is defined by a [BaseMatrix] whose container stores a memory slice.
pub type SliceMatrix<'a, Item, S> = Matrix<Item, BaseMatrix<Item, SliceContainer<'a, Item>, S>, S>;

/// Like [SliceMatrix] but with mutable access.
pub type SliceMatrixMut<'a, Item, S> =
    Matrix<Item, BaseMatrix<Item, SliceContainerMut<'a, Item>, S>, S>;

/// A dynamic matrix generic over the item type and the underlying layout.
pub type MatrixD<Item> = Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, Dynamic>, Dynamic>;

/// A matrix that provides a view onto part of another matrix.
pub type ViewMatrix<'a, Item, MatImpl, S> = Matrix<Item, MatrixView<'a, Item, MatImpl, S>, S>;

/// A matrix that provides a mutable view onto part of another matrix.
pub type ViewMatrixMut<'a, Item, MatImpl, S> = Matrix<Item, MatrixViewMut<'a, Item, MatImpl, S>, S>;

/// A fixed 2x2 matrix.
pub type Matrix22<Item> = Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 4>, Fixed2>, Fixed2>;

/// A fixed 3x3 matrix.
pub type Matrix33<Item> = Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 9>, Fixed3>, Fixed3>;

/// The basic tuple type defining a matrix. It is given as `(MatImpl, _, _, _, _)`.
/// The only relevant member is the first one `MatImpl`, an implementation type to which
/// all calls are forwarded. The other members are of type [PhantomData] and are necessary
/// to make the type dependent on the corresponding generic type parameter.
pub struct Matrix<Item, MatImpl, S>(MatImpl, PhantomData<Item>, PhantomData<S>)
where
    Item: Scalar,
    S: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, S>;
