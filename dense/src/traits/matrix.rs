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
    Layout, Size, SizeIdentifier, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::types::Scalar;
use crate::DefaultLayout;

#[derive(Debug, PartialEq)]
pub enum MatrixImplType {
    Base,
    Derived,
}

pub trait MatrixImplIdentifier {
    const MAT_IMPL: MatrixImplType;

    fn get_mat_impl_type(&self) -> MatrixImplType {
        Self::MAT_IMPL
    }
}

/// Combined trait for basic matrix properties. See [crate::traits::matrix]
/// for details.
pub trait MatrixImplTrait<Item: Scalar, S: SizeIdentifier>:
    UnsafeRandomAccessByValue<Item = Item>
    + Layout<Impl = DefaultLayout>
    + MatrixImplIdentifier
    + Size<S = S>
{
}

/// Extended Matrix trait if access by reference is possible.
pub trait MatrixImplTraitAccessByRef<Item: Scalar, S: SizeIdentifier>:
    UnsafeRandomAccessByRef<Item = Item> + MatrixImplTrait<Item, S>
{
}

/// Combined trait for mutable matrices. See [crate::traits::matrix] for details.
pub trait MatrixImplTraitMut<Item: Scalar, S: SizeIdentifier>:
    UnsafeRandomAccessMut<Item = Item> + MatrixImplTrait<Item, S>
{
}

impl<
        Item: Scalar,
        S: SizeIdentifier,
        Mat: UnsafeRandomAccessByValue<Item = Item>
            + Layout<Impl = DefaultLayout>
            + Size<S = S>
            + MatrixImplIdentifier,
    > MatrixImplTrait<Item, S> for Mat
{
}

impl<
        Item: Scalar,
        S: SizeIdentifier,
        Mat: UnsafeRandomAccessByRef<Item = Item> + MatrixImplTrait<Item, S>,
    > MatrixImplTraitAccessByRef<Item, S> for Mat
{
}

impl<
        Item: Scalar,
        S: SizeIdentifier,
        Mat: MatrixImplTrait<Item, S> + UnsafeRandomAccessMut<Item = Item>,
    > MatrixImplTraitMut<Item, S> for Mat
{
}
