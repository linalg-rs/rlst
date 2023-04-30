//! Definition of the main matrix type.
//!
//! [Matrix] is a basic type that provides an interface to dealing
//! with generic matrices. The only member of this type is an implementation
//! to which all calls are forwarded. Implementations can represent more basic
//! types, addition operations on matrices, scalar multiplications, or other
//! types of implementations. The only condition is that the implementation itself
//! implements [MatrixTrait] or [MatrixTraitMut].
//! A matrix is generic over the following parameters:
//! - `Item`. Implements the [Scalar] trait and represents the underlying scalar type
//!           of the matrix.
//! - `MatImpl`. The actual implementation of the matrix. It must itself implement the
//!              trait [MatrixTrait] or [MatrixTraitMut] depending on whether mutable access
//!              is required.
//! - `L`. A given type that implements the [LayoutType] trait and specifies the memory layout
//!        of the matrix.
//! - `RS`. A type that implements [SizeType] and specifies whether the row dimension is known
//!         at compile time or dynamically at runtime.
//! - `CS`. A type that implements [SizeType]  and specifies whether the column dimension is
//!         known at compile time or dynamically at runtime.

pub mod common_impl;
pub mod constructors;
pub mod iterators;
pub mod matrix_slices;
pub mod random;

use crate::base_matrix::BaseMatrix;
use crate::data_container::{ArrayContainer, SliceContainer, SliceContainerMut, VectorContainer};
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::Scalar;
use std::marker::PhantomData;

/// A [RefMat] is a matrix whose implementation is a reference to another matrix.
/// This is used to convert a reference to a matrix to an owned matrix whose implementation
/// is a reference to matrix.
pub type RefMat<'a, Item, MatImpl, RS, CS> =
    Matrix<Item, MatrixRef<'a, Item, MatImpl, RS, CS>, RS, CS>;

/// This type represents a generic matrix that depends on a [BaseMatrix] type.
pub type GenericBaseMatrix<Item, Data, RS, CS> =
    Matrix<Item, BaseMatrix<Item, Data, RS, CS>, RS, CS>;

/// A [SliceMatrix] is defined by a [BaseMatrix] whose container stores a memory slice.
pub type SliceMatrix<'a, Item, RS, CS> =
    Matrix<Item, BaseMatrix<Item, SliceContainer<'a, Item>, RS, CS>, RS, CS>;

/// Like [SliceMatrix] but with mutable access.
pub type SliceMatrixMut<'a, Item, RS, CS> =
    Matrix<Item, BaseMatrix<Item, SliceContainerMut<'a, Item>, RS, CS>, RS, CS>;

/// A dynamic matrix generic over the item type and the underlying layout.
pub type MatrixD<Item> =
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, Dynamic, Dynamic>, Dynamic, Dynamic>;

/// A dynamic column vector. This means that the row dimension is dynamic and the column
/// dimension is [Fixed1].
pub type ColumnVectorD<Item> = GenericBaseMatrix<Item, VectorContainer<Item>, Dynamic, Fixed1>;

/// A dynamic row vector. This means that the column dimension is dynamic and the row dimension
/// is [Fixed1].
pub type RowVectorD<Item> = GenericBaseMatrix<Item, VectorContainer<Item>, Fixed1, Dynamic>;

/// A fixed 2x2 matrix.
pub type Matrix22<Item> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 4>, Fixed2, Fixed2>, Fixed2, Fixed2>;

/// A fixed 3x3 matrix.
pub type Matrix33<Item> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 9>, Fixed3, Fixed3>, Fixed3, Fixed3>;

/// A fixed 3x2 matrix.
pub type Matrix32<Item> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 6>, Fixed3, Fixed2>, Fixed3, Fixed2>;

/// A fixed 2x3 matrix.
pub type Matrix23<Item> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 6>, Fixed2, Fixed3>, Fixed2, Fixed3>;

/// The basic tuple type defining a matrix. It is given as `(MatImpl, _, _, _, _)`.
/// The only relevant member is the first one `MatImpl`, an implementation type to which
/// all calls are forwarded. The other members are of type [PhantomData] and are necessary
/// to make the type dependent on the corresponding generic type parameter.
pub struct Matrix<Item, MatImpl, RS, CS>(
    MatImpl,
    PhantomData<Item>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: Scalar,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixImplTrait<Item, RS, CS>;
