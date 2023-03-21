//! Matrix trait
//!
//! [MatrixTrait] and its mutable counterpart [MatrixTraitMut] are
//! traits that define a matrix. [MatrixTrait] is automatically implemented
//! if the following traits are available.
//! - [RandomAccess]. Provides an interface to access matrix elements.
//! - [Layout]. Provides an interface to obtain layout information for the matrix.
//! - [SizeType]. Specifies whether the row/column dimension is known at compile time
//!               or specified at runtime.
//!
//! [MatrixTraitMut] additionally depends on the trait [RandomAccessMut] to provide
//! mutable access to matrix elements.
use crate::traits::{Layout, LayoutType, RandomAccess, RandomAccessMut, SizeIdentifier, SizeType};
use crate::types::Scalar;

/// Combined trait for basic matrix properties. See [crate::traits::matrix]
/// for details.
pub trait MatrixTrait<Item: Scalar, L: LayoutType, RS: SizeIdentifier, CS: SizeIdentifier>:
    RandomAccess<Item = Item> + Layout<Impl = L> + SizeType<R = RS, C = CS>
{
}

/// Combined trait for mutable matrices. See [crate::traits::matrix] for details.
pub trait MatrixTraitMut<Item: Scalar, L: LayoutType, RS: SizeIdentifier, CS: SizeIdentifier>:
    RandomAccessMut<Item = Item> + MatrixTrait<Item, L, RS, CS>
{
}

impl<
        Item: Scalar,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Mat: RandomAccess<Item = Item> + Layout<Impl = L> + SizeType<R = RS, C = CS>,
    > MatrixTrait<Item, L, RS, CS> for Mat
{
}

impl<
        Item: Scalar,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Mat: MatrixTrait<Item, L, RS, CS> + RandomAccessMut<Item = Item>,
    > MatrixTraitMut<Item, L, RS, CS> for Mat
{
}
