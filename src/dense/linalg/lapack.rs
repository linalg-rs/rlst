//! Lapack interface for linear algebra operations.

pub mod inverse;
pub mod lu;

use inverse::LapackInverse;

use crate::{
    dense::{
        array::Array,
        traits::{RawAccess, RawAccessMut, Shape, Stride},
    },
    BaseItem,
};

/// Return a triple (m, n, lda) for the Lapack interface.
pub fn lapack_dims<ArrayImpl>(arr: &Array<ArrayImpl, 2>) -> (i32, i32, i32)
where
    ArrayImpl: Shape<2> + Stride<2>,
{
    let shape = arr.shape();
    let stride = arr.stride();

    let m = shape[0] as i32;
    let n = shape[1] as i32;
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );

    let lda = arr.stride()[1] as i32;

    (m, n, lda)
}

/// A wrapper for LAPACK operations on a non-mutable 2D array
pub struct LapackWrapper<'a, Item> {
    data: &'a [Item],
    m: i32,
    n: i32,
    lda: i32,
}

/// A wrapper for LAPACK operations on a mutable 2D array.
pub struct LapackWrapperMut<'a, Item> {
    data: &'a mut [Item],
    m: i32,
    n: i32,
    lda: i32,
}

pub trait LapackOperationsMut
where
    for<'a> LapackWrapperMut<'a, Self::Item>: LapackInverse,
{
    /// The item type contained in the array.
    type Item;

    /// Interface to LAPACK operations.
    fn lapack_mut(&mut self) -> LapackWrapperMut<'_, Self::Item>;
}

impl<Item, ArrayImpl> LapackOperationsMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Item>,
    for<'a> LapackWrapperMut<'a, Item>: LapackInverse,
{
    type Item = Item;

    fn lapack_mut(&mut self) -> LapackWrapperMut<'_, Self::Item> {
        let (m, n, lda) = lapack_dims(self);
        LapackWrapperMut {
            data: self.data_mut(),
            m,
            n,
            lda,
        }
    }
}
