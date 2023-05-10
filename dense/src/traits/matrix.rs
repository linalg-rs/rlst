//! Matrix Implementation trait
//!
//! [MatrixImplTrait] and its mutable counterpart [MatrixImplTraitMut] are
//! traits that define matrix implementations. The [MatrixImplTrait] is auto-implemented
//! for any implementation that supports [UnsafeRandomAccessByValue], [Layout] and [SizeType].
//! If the matrix elements are associated with a physical memory location one can implement
//! [UnsafeRandomAccessByRef]. If also [MatrixImplTrait] is implemented the trait
//! [MatrixImplTraitAccessByRef] is then auto-implemented. This marks matrix implementations
//! who allow access by reference. The trait[MatrixImplTraitMut] marks an implementation that
//! allows mutation of matrix elements. It is auto-implemented if [MatrixImplTrait] and
//! [UnsafeRandomAccessMut] are implemented.
//!
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
