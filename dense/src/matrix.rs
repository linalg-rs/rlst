//! Definition of the main matrix type.
//!
//! [Matrix] is a basic type that provides an interface to dealing
//! with generic matrices. The only member of this type is an implementation
//! to which all calls are forwarded. Implementations can represent more basic
//! types, addition operations on matrices, scalar multiplications, or other
//! types of implementations. The only condition is that the implementation itself
//! implements [MatrixTrait] or [MatrixTraitMut].
//! A matrix is generic over the following parameters:
//! - `Item`. Implements the [HScalar] trait and represents the underlying scalar type
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

pub mod base_methods;
pub mod common_impl;
pub mod constructors;
pub mod matrix_slices;
pub mod random;

use crate::base_matrix::BaseMatrix;
use crate::data_container::{
    ArrayContainer, DataContainer, SliceContainer, SliceContainerMut, VectorContainer,
};
use crate::layouts::*;
use crate::matrix_ref::MatrixRef;
use crate::traits::*;
use crate::types::HScalar;
use std::marker::PhantomData;

/// A [RefMat] is a matrix whose implementation is a reference to another matrix.
/// This is used to convert a reference to a matrix to an owned matrix whose implementation
/// is a reference to matrix.
pub type RefMat<'a, Item, MatImpl, L, RS, CS> =
    Matrix<Item, MatrixRef<'a, Item, MatImpl, L, RS, CS>, L, RS, CS>;

/// This type represents a generic matrix that depends on a [BaseMatrix] type.
pub type GenericBaseMatrix<Item, L, Data, RS, CS> =
    Matrix<Item, BaseMatrix<Item, Data, L, RS, CS>, L, RS, CS>;

/// Similar to a [GenericBaseMatrix] but with mutable access.
pub type GenericBaseMatrixMut<Item, L, Data, RS, CS> =
    Matrix<Item, BaseMatrix<Item, Data, L, RS, CS>, L, RS, CS>;

/// A [SliceMatrix] is defined by a [BaseMatrix] whose container stores a memory slice.
pub type SliceMatrix<'a, Item, L, RS, CS> =
    Matrix<Item, BaseMatrix<Item, SliceContainer<'a, Item>, L, RS, CS>, L, RS, CS>;

/// Like [SliceMatrix] but with mutable access.
pub type SliceMatrixMut<'a, Item, L, RS, CS> =
    Matrix<Item, BaseMatrix<Item, SliceContainerMut<'a, Item>, L, RS, CS>, L, RS, CS>;

/// A dynamic matrix generic over the item type and the underlying layout.
pub type MatrixD<Item, L> =
    Matrix<Item, BaseMatrix<Item, VectorContainer<Item>, L, Dynamic, Dynamic>, L, Dynamic, Dynamic>;

/// A dynamic upper triangular matrix.
pub type UpperTriangularMatrix<Item> = Matrix<
    Item,
    BaseMatrix<Item, VectorContainer<Item>, UpperTriangular, Dynamic, Dynamic>,
    UpperTriangular,
    Dynamic,
    Dynamic,
>;

/// A dynamic column vector. This means that the row dimension is dynamic and the column
/// dimension is [Fixed1].
pub type ColumnVectorD<Item> =
    GenericBaseMatrixMut<Item, ColumnVector, VectorContainer<Item>, Dynamic, Fixed1>;

/// A dynamic row vector. This means that the column dimension is dynamic and the row dimension
/// is [Fixed1].
pub type RowVectorD<Item> =
    GenericBaseMatrixMut<Item, RowVector, VectorContainer<Item>, Fixed1, Dynamic>;

/// A fixed 2x2 matrix.
pub type Matrix22<Item, L> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 4>, L, Fixed2, Fixed2>, L, Fixed2, Fixed2>;

/// A fixed 3x3 matrix.
pub type Matrix33<Item, L> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 9>, L, Fixed3, Fixed3>, L, Fixed3, Fixed3>;

/// A fixed 3x2 matrix.
pub type Matrix32<Item, L> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 6>, L, Fixed3, Fixed2>, L, Fixed3, Fixed2>;

/// A fixed 2x3 matrix.
pub type Matrix23<Item, L> =
    Matrix<Item, BaseMatrix<Item, ArrayContainer<Item, 6>, L, Fixed2, Fixed3>, L, Fixed2, Fixed3>;

/// The basic tuple type defining a matrix. It is given as `(MatImpl, _, _, _, _)`.
/// The only relevant member is the first one `MatImpl`, an implementation type to which
/// all calls are forwarded. The other members are of type [PhantomData] and are necessary
/// to make the type dependent on the corresponding generic type parameter.
pub struct Matrix<Item, MatImpl, L, RS, CS>(
    MatImpl,
    PhantomData<Item>,
    PhantomData<L>,
    PhantomData<RS>,
    PhantomData<CS>,
)
where
    Item: HScalar,
    L: LayoutType,
    RS: SizeIdentifier,
    CS: SizeIdentifier,
    MatImpl: MatrixTrait<Item, L, RS, CS>;

impl<
        Item: HScalar,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        MatImpl: MatrixTrait<Item, L, RS, CS>,
    > Matrix<Item, MatImpl, L, RS, CS>
{
    pub fn new(mat: MatImpl) -> Self {
        Self(mat, PhantomData, PhantomData, PhantomData, PhantomData)
    }

    pub fn from_ref<'a>(
        mat: &'a Matrix<Item, MatImpl, L, RS, CS>,
    ) -> RefMat<'a, Item, MatImpl, L, RS, CS> {
        RefMat::new(MatrixRef::new(mat))
    }
}

impl<
        Item: HScalar,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainer<Item = Item>,
    > Matrix<Item, BaseMatrix<Item, Data, L, RS, CS>, L, RS, CS>
{
    pub fn from_data(data: Data, layout: L) -> Self {
        Self::new(BaseMatrix::<Item, Data, L, RS, CS>::new(data, layout))
    }
}
