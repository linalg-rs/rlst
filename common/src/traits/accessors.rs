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

use crate::traits::properties::{NumberOfElements, Shape};
use crate::types::Scalar;

/// This trait provides unsafe access by value to the underlying data.
pub trait UnsafeRandomAccessByValue {
    type Item: Scalar;

    /// Return the element at position (`row`, `col`).
    ///
    /// # Safety
    /// `row` and `col` must not be out of bounds.
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item;

    /// Return the element at position `index` in one-dimensional numbering.
    ///
    /// # Safety
    /// `row` and `col` must not be out of bounds.
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item;
}

/// This trait provides unsafe access by reference to the underlying data.
pub trait UnsafeRandomAccessByRef {
    type Item: Scalar;

    /// Return a mutable reference to the element at position (`row`, `col`).
    ///
    /// # Safety
    /// `row` and `col` must not be out of bounds.
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item;

    /// Return a mutable reference at position `index` in one-dimensional numbering.
    ///
    /// # Safety
    /// `row` and `col` must not be out of bounds.
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data.
pub trait UnsafeRandomAccessMut {
    type Item: Scalar;

    /// Return a mutable reference to the element at position (`row`, `col`).
    ///
    /// # Safety
    /// `row` and `col` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Item;

    /// Return a mutable reference at position `index` in one-dimensional numbering.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// This trait provides bounds checked access to the underlying data by value.
pub trait RandomAccessByValue: UnsafeRandomAccessByValue {
    /// Return the element at position (`row`, `col`).
    fn get_value(&self, row: usize, col: usize) -> Option<Self::Item>;

    /// Return the element at position `index` in one-dimensional numbering.
    fn get1d_value(&self, elem: usize) -> Option<Self::Item>;
}

/// This trait provides bounds checked access to the underlying data by reference.
pub trait RandomAccessByRef: UnsafeRandomAccessByRef {
    /// Return a reference to the element at position (`row`, `col`).
    fn get(&self, row: usize, col: usize) -> Option<&Self::Item>;

    /// Return a reference at position `index` in one-dimensional numbering.
    fn get1d(&self, elem: usize) -> Option<&Self::Item>;
}

/// This trait provides bounds checked mutable access to the underlying data.
pub trait RandomAccessMut: UnsafeRandomAccessMut {
    /// Return a mutable reference to the element at position (`row`, `col`).
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item>;

    /// Return a mutable reference at position `index` in one-dimensional numbering.
    fn get1d_mut(&mut self, elem: usize) -> Option<&mut Self::Item>;
}

/// Get raw access to the underlying data.
pub trait RawAccess {
    type T: Scalar;

    /// Get a raw pointer to the data.
    fn get_pointer(&self) -> *const Self::T;

    /// Get a data slice in the half-open interval `[first, last)`,
    /// which includes `first` and excludes `last`.
    fn get_slice(&self, first: usize, last: usize) -> &[Self::T];

    /// Get a slice of the whole data.
    fn data(&self) -> &[Self::T];
}

/// Get mutable raw access to the underlying data.
pub trait RawAccessMut: RawAccess {
    /// Get a mutable raw pointer to the data.
    fn get_pointer_mut(&mut self) -> *mut Self::T;

    /// Get a mutable data slice in the half-open interval `[first, last)`,
    /// which includes `first` and excludes `last`.
    fn get_slice_mut(&mut self, first: usize, last: usize) -> &mut [Self::T];

    /// Get a mutable slice of the whole data.
    fn data_mut(&mut self) -> &mut [Self::T];
}

/// Check if `row` and `col` are valid with respect to the shape `shape`.
#[inline]
fn check_dimension(row: usize, col: usize, shape: (usize, usize)) -> bool {
    row < shape.0 && col < shape.1
}

/// Check if `elem` is smaller than `nelems`.
#[inline]
fn check_dimension1d(elem: usize, nelems: usize) -> bool {
    elem < nelems
}

impl<Item: Scalar, Mat: UnsafeRandomAccessByValue<Item = Item> + Shape + NumberOfElements>
    RandomAccessByValue for Mat
{
    fn get_value(&self, row: usize, col: usize) -> Option<Self::Item> {
        if check_dimension(row, col, self.shape()) {
            Some(unsafe { self.get_value_unchecked(row, col) })
        } else {
            None
        }
    }

    fn get1d_value(&self, elem: usize) -> Option<Self::Item> {
        if check_dimension1d(elem, self.number_of_elements()) {
            Some(unsafe { self.get1d_value_unchecked(elem) })
        } else {
            None
        }
    }
}

impl<Item: Scalar, Mat: UnsafeRandomAccessMut<Item = Item> + Shape + NumberOfElements>
    RandomAccessMut for Mat
{
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Item> {
        if check_dimension(row, col, self.shape()) {
            unsafe { Some(self.get_unchecked_mut(row, col)) }
        } else {
            None
        }
    }

    fn get1d_mut(&mut self, elem: usize) -> Option<&mut Self::Item> {
        if check_dimension1d(elem, self.number_of_elements()) {
            unsafe { Some(self.get1d_unchecked_mut(elem)) }
        } else {
            None
        }
    }
}

impl<Item: Scalar, Mat: UnsafeRandomAccessByRef<Item = Item> + Shape + NumberOfElements>
    RandomAccessByRef for Mat
{
    fn get(&self, row: usize, col: usize) -> Option<&Self::Item> {
        if check_dimension(row, col, self.shape()) {
            unsafe { Some(self.get_unchecked(row, col)) }
        } else {
            None
        }
    }

    fn get1d(&self, elem: usize) -> Option<&Self::Item> {
        if check_dimension1d(elem, self.number_of_elements()) {
            unsafe { Some(self.get1d_unchecked(elem)) }
        } else {
            None
        }
    }
}
