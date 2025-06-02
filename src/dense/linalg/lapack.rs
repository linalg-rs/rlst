//! Lapack interface for linear algebra operations.

pub mod inverse;
pub mod lu;

use inverse::LapackInverse;

use crate::dense::{
    array::Array,
    traits::{RawAccess, RawAccessMut, Shape, Stride},
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

/// A wrapper for LAPACK operations on a 2D array.
pub struct LapackWrapper<Item, ArrayImpl> {
    arr: Array<ArrayImpl, 2>,
    m: i32,
    n: i32,
    lda: i32,
    _marker: std::marker::PhantomData<Item>,
}

impl<Item, ArrayImpl> LapackWrapper<Item, ArrayImpl>
where
    ArrayImpl: Shape<2> + Stride<2>,
{
    /// Create a new `LapackWrapper` from an array.
    pub fn new(arr: Array<ArrayImpl, 2>) -> Self {
        let (m, n, lda) = lapack_dims(&arr);
        Self {
            arr,
            m,
            n,
            lda,
            _marker: std::marker::PhantomData,
        }
    }
}
