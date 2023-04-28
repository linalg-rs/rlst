//! Matrix trait
//!
//! [MatrixTrait] and its mutable counterpart [MatrixTraitMut] are
//! traits that define a matrix. [MatrixTrait] is automatically implemented
//! if the following traits are available.
//! - [RandomAccessByValue]. Provides an interface to access matrix elements.
//! - [Layout]. Provides an interface to obtain layout information for the matrix.
//! - [SizeType]. Specifies whether the row/column dimension is known at compile time
//!               or specified at runtime.
//!
//! [MatrixTraitMut] additionally depends on the trait [RandomAccessMut] to provide
//! mutable access to matrix elements.
//!
//! In order to support standard index notation for Matrix elements a matrix
//! needs to additionally define the trait [RandomAccessByRef]. This then
//! auto-implements the trait [MatrixTraitAccessByRef].
use crate::traits::{
    Layout, SizeIdentifier, SizeType, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::types::Scalar;
use crate::DefaultLayout;

/// Combined trait for basic matrix properties. See [crate::traits::matrix]
/// for details.
pub trait MatrixImplTrait<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier>:
    UnsafeRandomAccessByValue<Item = Item> + Layout<Impl = DefaultLayout> + SizeType<R = RS, C = CS>
{
}

/// Extended Matrix trait if access by reference is possible.
pub trait MatrixImplTraitAccessByRef<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier>:
    UnsafeRandomAccessByRef<Item = Item> + MatrixImplTrait<Item, RS, CS>
{
}

/// Combined trait for mutable matrices. See [crate::traits::matrix] for details.
pub trait MatrixImplTraitMut<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier>:
    UnsafeRandomAccessMut<Item = Item> + MatrixImplTrait<Item, RS, CS>
{
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Mat: UnsafeRandomAccessByValue<Item = Item>
            + Layout<Impl = DefaultLayout>
            + SizeType<R = RS, C = CS>,
    > MatrixImplTrait<Item, RS, CS> for Mat
{
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Mat: UnsafeRandomAccessByRef<Item = Item> + MatrixImplTrait<Item, RS, CS>,
    > MatrixImplTraitAccessByRef<Item, RS, CS> for Mat
{
}

impl<
        Item: Scalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Mat: MatrixImplTrait<Item, RS, CS> + UnsafeRandomAccessMut<Item = Item>,
    > MatrixImplTraitMut<Item, RS, CS> for Mat
{
}
