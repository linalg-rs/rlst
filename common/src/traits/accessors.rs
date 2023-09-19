//! Traits for access to matrix data.
//!
//! Each random access trait has a two-dimensional and a one-dimensional access method,
//! namely `get` and `get1d` (together with their mutable and unsafe variants).
//! The two-dimensional access takes a row and a column and returns the corresponding
//! matrix element. The one-dimensional access takes a single `index` parameter that
//! iterates through the matrix elements.
//!
//! If the [`crate::traits::properties::Shape`] and [`crate::traits::properties::NumberOfElements`]
//! traits are implemented on top of [`UnsafeRandomAccessByValue`], [`UnsafeRandomAccessByRef`]
//! and [`UnsafeRandomAccessMut`] then the
//! corresponding bounds-checked traits [`RandomAccessByValue`], [`RandomAccessByRef`] and
//! [`RandomAccessMut`] are auto-implemented.
//!
//! To get raw access to the underlying data use the [`RawAccess`] and [`RawAccessMut`] traits.

use crate::traits::properties::Shape;
use crate::types::{DataChunk, Scalar};

/// This trait provides unsafe access by value to the underlying data.
pub trait UnsafeRandomAccessByValue<const NDIM: usize> {
    type Item: Scalar;

    /// Return the element at position determined by `indices`.
    ///
    /// # Safety
    /// `indices` must not be out of bounds.
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item;
}

/// This trait provides unsafe access by reference to the underlying data.
pub trait UnsafeRandomAccessByRef<const NDIM: usize> {
    type Item: Scalar;

    /// Return a mutable reference to the element at position determined by `indices`.
    ///
    /// # Safety
    /// `indices` must not be out of bounds.
    unsafe fn get_unchecked(&self, indices: [usize; NDIM]) -> &Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data.
pub trait UnsafeRandomAccessMut<const NDIM: usize> {
    type Item: Scalar;

    /// Return a mutable reference to the element at position determined by `indices`.
    ///
    /// # Safety
    /// `indices` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, indices: [usize; NDIM]) -> &mut Self::Item;
}

/// This trait provides bounds checked access to the underlying data by value.
pub trait RandomAccessByValue<const NDIM: usize>: UnsafeRandomAccessByValue<NDIM> {
    /// Return the element at position determined by `indices`.
    fn get_value(&self, indices: [usize; NDIM]) -> Option<Self::Item>;
}

/// This trait provides bounds checked access to the underlying data by reference.
pub trait RandomAccessByRef<const NDIM: usize>: UnsafeRandomAccessByRef<NDIM> {
    /// Return a reference to the element at position determined by `indices`.
    fn get(&self, indices: [usize; NDIM]) -> Option<&Self::Item>;
}

/// This trait provides bounds checked mutable access to the underlying data.
pub trait RandomAccessMut<const NDIM: usize>: UnsafeRandomAccessMut<NDIM> {
    /// Return a mutable reference to the element at position determined by `indices`.
    fn get_mut(&mut self, indices: [usize; NDIM]) -> Option<&mut Self::Item>;
}

/// Return chunks of data of size N;
pub trait ChunkedAccess<const N: usize> {
    type Item: Scalar;
    fn get_chunk(&self, chunk_index: usize) -> Option<DataChunk<Self::Item, N>>;
}

/// Get raw access to the underlying data.
pub trait RawAccess {
    type T: Scalar;

    /// Get a slice of the whole data.
    fn data(&self) -> &[Self::T];
}

/// Get mutable raw access to the underlying data.
pub trait RawAccessMut: RawAccess {
    /// Get a mutable slice of the whole data.
    fn data_mut(&mut self) -> &mut [Self::T];
}

/// Check if `indices` not out of bounds with respect to `shape`.
#[inline]
fn check_dimension<const NDIM: usize>(indices: [usize; NDIM], shape: [usize; NDIM]) -> bool {
    indices
        .iter()
        .zip(shape.iter())
        .fold(true, |acc, (ind, s)| acc && (ind < s))
}

impl<
        Item: Scalar,
        Mat: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > RandomAccessByValue<NDIM> for Mat
{
    fn get_value(&self, indices: [usize; NDIM]) -> Option<Self::Item> {
        if check_dimension(indices, self.shape()) {
            Some(unsafe { self.get_value_unchecked(indices) })
        } else {
            None
        }
    }
}

impl<
        Item: Scalar,
        Mat: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > RandomAccessMut<NDIM> for Mat
{
    fn get_mut(&mut self, indices: [usize; NDIM]) -> Option<&mut Self::Item> {
        if check_dimension(indices, self.shape()) {
            unsafe { Some(self.get_unchecked_mut(indices)) }
        } else {
            None
        }
    }
}

impl<
        Item: Scalar,
        Mat: UnsafeRandomAccessByRef<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > RandomAccessByRef<NDIM> for Mat
{
    fn get(&self, indices: [usize; NDIM]) -> Option<&Self::Item> {
        if check_dimension(indices, self.shape()) {
            unsafe { Some(self.get_unchecked(indices)) }
        } else {
            None
        }
    }
}
