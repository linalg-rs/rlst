//! Lapack interface for linear algebra operations.

pub mod inverse;
pub mod lu;

use crate::dense::{
    array::Array,
    traits::{RawAccess, RawAccessMut, Shape, Stride},
};

/// Return true if stride is column major as required by Lapack.
pub fn assert_lapack_stride(stride: [usize; 2]) {
    assert_eq!(
        stride[0], 1,
        "Incorrect stride for Lapack. Stride[0] is {} but expected 1.",
        stride[0]
    );
}

/// A structure that represents a 2D array in a format suitable for Lapack operations.
pub struct LapackArray<'a, Item> {
    /// The underlying data.
    pub data: &'a [Item],
    /// The number of rows.
    pub m: i32,
    /// The number of columns.
    pub n: i32,
    /// The leading dimension (stride in the second dimension).
    pub lda: i32,
}

/// A structure that represents a 2D array in a format suitable for Lapack operations.
pub struct LapackArrayMut<'a, Item> {
    /// The underlying data.
    pub data: &'a mut [Item],
    /// The number of rows.
    pub m: i32,
    /// The number of columns.
    pub n: i32,
    /// The leading dimension (stride in the second dimension).
    pub lda: i32,
}

impl<'a, Item> LapackArray<'a, Item> {
    /// Create a new LapackArray from the given array.
    pub fn new<ArrayImpl: Stride<2> + Shape<2> + RawAccess<Item = Item>>(
        arr: &'a Array<ArrayImpl, 2>,
    ) -> Self {
        assert_lapack_stride(arr.stride());
        Self {
            data: arr.data(),
            m: arr.shape()[0] as i32,
            n: arr.shape()[1] as i32,
            lda: arr.stride()[1] as i32,
        }
    }
}

impl<'a, Item> LapackArrayMut<'a, Item> {
    /// Create a new LapackArray from the given array.
    pub fn new<ArrayImpl: Stride<2> + Shape<2> + RawAccessMut<Item = Item>>(
        arr: &'a mut Array<ArrayImpl, 2>,
    ) -> Self {
        assert_lapack_stride(arr.stride());
        let m = arr.shape()[0] as i32;
        let n = arr.shape()[1] as i32;
        let lda = arr.stride()[1] as i32;

        Self {
            data: arr.data_mut(),
            m,
            n,
            lda,
        }
    }
}

/// A trait for types are compatible with mutable Lapack operations.
pub trait LapackMut {
    /// The array implementation type.
    type Item;

    /// Convert the array into a LapackArray.
    fn lapack_mut(&mut self) -> LapackArrayMut<'_, Self::Item>;
}

///  A trait for types that are compatible with Lapack operations.
pub trait Lapack {
    /// The array implementation type.
    type Item;

    /// Convert the array into a LapackArray.
    fn lapack(&self) -> LapackArray<'_, Self::Item>;
}

impl<ArrayImpl> Lapack for Array<ArrayImpl, 2>
where
    ArrayImpl: Stride<2> + Shape<2> + RawAccess,
{
    type Item = ArrayImpl::Item;

    fn lapack(&self) -> LapackArray<'_, Self::Item> {
        LapackArray::new(self)
    }
}

impl<Item, ArrayImpl> LapackMut for Array<ArrayImpl, 2>
where
    ArrayImpl: Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
{
    type Item = ArrayImpl::Item;

    fn lapack_mut(&mut self) -> LapackArrayMut<'_, Self::Item> {
        LapackArrayMut::new(self)
    }
}
