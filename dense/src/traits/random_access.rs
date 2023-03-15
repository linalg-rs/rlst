//! Traits for random access of matrices
//!
//! The traits in this module define safe and unsafe random access
//! methods for a matrix. The user needs to implement `UnsafeRandomAccess`
//! and `UnsafeRandomAccessMut` (for mutable access).
//!
//! If additionally the [Layout](crate::traits::Layout) is implemented (aut-implemented
//! if the [LayoutType](crate::traits::LayoutType) trait is implemented), then
//! he corresponding safe traits `SafeRandomAccess` and
//! `SafeRandomAccessMut` are auto-implemented.
//!
//! Each trait provides a two-dimensional and a one-dimensional access method,
//! namely `get` and `get1d` (together with their mutable and unsafe variants).
//! The two-dimensional access takes a row and a column and returns the corresponding
//! matrix element. The one-dimensional access takes a single
//!
//! The one-dimensional access
//! takes a single `index` parameter that iterates through the matrix elements.
//! It is recommended to use the [convert_1d_raw](crate::traits::LayoutType::convert_1d_raw)
//! functions from that trait to implement `get1d` to ensure compatibility with
//! the memory layout defined in that trait.

use crate::traits::{Layout, LayoutType};
use crate::types::{HScalar, IndexType};

/// This trait provides unsafe access to the underlying data. See
/// [Random Access](crate::traits::random_access) for a description.
pub trait UnsafeRandomAccess {
    type Item: HScalar;

    /// Return the element at position (`row`, `col`).
    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> Self::Item;

    /// Return the element at position `index` in one-dimensional numbering.
    unsafe fn get1d_unchecked(&self, index: IndexType) -> Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data. See
/// [Random Access](crate::traits::random_access) for a description.
pub trait UnsafeRandomAccessMut {
    type Item: HScalar;

    /// Return a mutable reference to the element at position (`row`, `col`).
    unsafe fn get_unchecked_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::Item;

    /// Return a mutable reference at position `index` in one-dimensional numbering.
    unsafe fn get1d_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item;
}

/// This trait provides bounds checked access to the underlying data. See
/// [Random Access](crate::traits::random_access) for a description.
pub trait RandomAccess: UnsafeRandomAccess {
    /// Return the element at position (`row`, `col`).
    fn get(&self, row: usize, col: usize) -> Self::Item;

    /// Return the element at position `index` in one-dimensional numbering.
    fn get1d(&self, elem: usize) -> Self::Item;
}

/// This trait provides bounds checked mutable access to the underlying data. See
/// [Random Access](crate::traits::random_access) for a description.
pub trait RandomAccessMut: UnsafeRandomAccessMut {
    /// Return a mutable reference to the element at position (`row`, `col`).
    fn get_mut(&mut self, row: usize, col: usize) -> &mut Self::Item;

    /// Return a mutable reference at position `index` in one-dimensional numbering.
    fn get1d_mut(&mut self, elem: usize) -> &mut Self::Item;
}

/// Check that a given pair of `row` and `col` is not out of bounds for given dimension `dim`.
#[inline]
fn assert_dimension(row: IndexType, col: IndexType, dim: (IndexType, IndexType)) {
    assert!(
        row < dim.0,
        "row {} out of bounds (dim: {}, {}",
        row,
        dim.0,
        dim.1
    );
    assert!(
        col < dim.1,
        "col {} out of bounds (dim: {}, {}",
        col,
        dim.0,
        dim.1
    );
}

/// Check that a given `index` parameter is not out of bounds for `nelems` elements.
#[inline]
fn assert_dimension1d(elem: IndexType, nelems: IndexType) {
    assert!(
        elem < nelems,
        "elem {} out of bounds (nelems: {})",
        elem,
        nelems
    );
}

impl<Item: HScalar, Mat: UnsafeRandomAccess<Item = Item> + Layout> RandomAccess for Mat {
    fn get(&self, row: IndexType, col: IndexType) -> Self::Item {
        assert_dimension(row, col, self.layout().dim());
        unsafe { self.get_unchecked(row, col) }
    }

    fn get1d(&self, elem: IndexType) -> Self::Item {
        assert_dimension1d(elem, self.layout().number_of_elements());
        unsafe { self.get1d_unchecked(elem) }
    }
}

impl<Item: HScalar, Mat: UnsafeRandomAccessMut<Item = Item> + Layout> RandomAccessMut for Mat {
    fn get_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::Item {
        assert_dimension(row, col, self.layout().dim());
        unsafe { self.get_unchecked_mut(row, col) }
    }

    fn get1d_mut(&mut self, elem: IndexType) -> &mut Self::Item {
        assert_dimension1d(elem, self.layout().number_of_elements());
        unsafe { self.get1d_unchecked_mut(elem) }
    }
}
