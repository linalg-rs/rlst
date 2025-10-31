//! Accessor traits for n-dimensional array-like structures.
//!
//! The traits in this module provide indexed access to array-like structures
//! of the form `arr[multi_index]` with `multi_index` an array of type `[usize; NDIM]`.
//! `NDIM` is the number of dimensions in the array. For example `arr[[3, 4, 8]]` returns
//! the element at position `[3, 4, 8]`.
//!
//! If an array structure is associated with an underlying memory region one can return
//! references to the underlying elements. However, this may not always be the case. The traits
//! in this module therefore distinguish between access by reference and access by value. For access
//! by value the underlying value type needs to support the `Copy` trait.
//!
//! The traits in this module are available as unsafe variant and as safe variant.
//! If the unsafe variant and the [Shape] trait are implemented the safe variant
//! is automatically implemented by checking whether the provided multi index is within the shape bounds.
//!
//! The traits are also available as 1d index variants. This is mainly useful for the implementation of iterators.
//! By convention in this library a 1d index is always assumed to be derived from a column-major enumeration of the
//! elements. The 1d indexed traits are only available as unsafe variants.
//!
//! For direct memory access the traits [RawAccess] and [RawAccessMut] are provided. These return the raw memory slices
//! of underlying data.

use super::base_operations::{BaseItem, Shape};

/// This trait provides unsafe access by value to the underlying data.
pub trait UnsafeRandomAccessByValue<const NDIM: usize>: BaseItem {
    /// Return the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_value_unchecked(&self, multi_index: [usize; NDIM]) -> Self::Item;
}

/// This trait provides unsafe access by value to the underlying data using a 1d index.
///
/// 1d indexing is always in column-major order.
///
/// # Safety
/// `index` must not be out of bounds.
pub trait UnsafeRandom1DAccessByValue: BaseItem {
    /// Return the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_value_1d_unchecked(&self, index: usize) -> Self::Item;
}

/// This trait provides unsafe access by reference to the underlying data.
pub trait UnsafeRandomAccessByRef<const NDIM: usize>: BaseItem {
    /// Return a reference to the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_unchecked(&self, multi_index: [usize; NDIM]) -> &Self::Item;
}

/// This trait provides unsafe access by reference to the underlying data using a 1d index.
pub trait UnsafeRandom1DAccessByRef: BaseItem {
    /// Return a reference to the element at position determined by `index`.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_1d_unchecked(&self, index: usize) -> &Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data.
pub trait UnsafeRandomAccessMut<const NDIM: usize>: BaseItem {
    /// Return a mutable reference to the element at position determined by `multi_index`.
    ///
    /// # Safety
    /// `multi_index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, multi_index: [usize; NDIM]) -> &mut Self::Item;
}

/// This trait provides unsafe mutable access to the underlying data using a 1D index.
pub trait UnsafeRandom1DAccessMut: BaseItem {
    /// Return a mutable reference to the element at position determined by `index`.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// This trait provides bounds checked access to the underlying data by value.
pub trait RandomAccessByValue<const NDIM: usize>: BaseItem {
    /// Return the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get_value(&self, multi_index: [usize; NDIM]) -> Option<Self::Item>;
}

/// This trait provides bounds checked access to the underlying data by reference.
pub trait RandomAccessByRef<const NDIM: usize>: BaseItem {
    /// Return a reference to the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get(&self, multi_index: [usize; NDIM]) -> Option<&Self::Item>;
}

/// This trait provides bounds checked mutable access to the underlying data.
pub trait RandomAccessMut<const NDIM: usize>: BaseItem {
    /// Return a mutable reference to the element at position determined by `multi_index`.
    /// If the value is out of bounds `None` is returned.
    fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut Self::Item>;
}

/// Get raw access to the underlying data.
pub trait RawAccess: BaseItem {
    /// Get a slice of the data.
    fn data(&self) -> &[Self::Item];
}

/// Get mutable raw access to the underlying data.
pub trait RawAccessMut: RawAccess {
    /// Get a mutable slice of the whole data.
    fn data_mut(&mut self) -> &mut [Self::Item];
}

/// Check if `multi_index` not out of bounds with respect to `shape`.
#[inline]
fn check_dimension<const NDIM: usize>(multi_index: [usize; NDIM], shape: [usize; NDIM]) -> bool {
    multi_index
        .iter()
        .zip(shape.iter())
        .fold(true, |acc, (ind, s)| acc && (ind < s))
}

impl<Mat: UnsafeRandomAccessByValue<NDIM> + Shape<NDIM>, const NDIM: usize>
    RandomAccessByValue<NDIM> for Mat
{
    fn get_value(&self, multi_index: [usize; NDIM]) -> Option<Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            Some(unsafe { self.get_value_unchecked(multi_index) })
        } else {
            None
        }
    }
}

impl<Item, Mat: UnsafeRandomAccessMut<NDIM, Item = Item> + Shape<NDIM>, const NDIM: usize>
    RandomAccessMut<NDIM> for Mat
{
    fn get_mut(&mut self, multi_index: [usize; NDIM]) -> Option<&mut Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            unsafe { Some(self.get_unchecked_mut(multi_index)) }
        } else {
            None
        }
    }
}

impl<Item, Mat: UnsafeRandomAccessByRef<NDIM, Item = Item> + Shape<NDIM>, const NDIM: usize>
    RandomAccessByRef<NDIM> for Mat
{
    fn get(&self, multi_index: [usize; NDIM]) -> Option<&Self::Item> {
        if check_dimension(multi_index, self.shape()) {
            unsafe { Some(self.get_unchecked(multi_index)) }
        } else {
            None
        }
    }
}
