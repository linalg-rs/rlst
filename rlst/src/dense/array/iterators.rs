//! Various iterator implementations.

use crate::dense::array::{Array, Shape};
use crate::dense::layout::convert_1d_nd_from_shape;
use crate::traits::accessors::{
    UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::traits::{
    base_operations::{BaseItem, Len},
    iterators::AsMultiIndex,
};
use crate::{
    AsOwnedRefType, AsOwnedRefTypeMut, UnsafeRandom1DAccessByRef, UnsafeRandomAccessByRef,
};

use super::reference::{self, ArrayRefMut};
use super::slice::ArraySlice;

/// Default column major iterator.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIteratorByValue<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// Default column major iterator by reference.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIteratorByRef<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// Mutable default iterator. Like [ArrayDefaultIterator] but with mutable access.
pub struct ArrayDefaultIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

/// A multi-index iterator returns the corrent element and the corresponding multi-index.
pub struct MultiIndexIterator<I, const NDIM: usize> {
    shape: [usize; NDIM],
    iter: I,
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> Iterator
    for MultiIndexIterator<I, NDIM>
{
    type Item = ([usize; NDIM], T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((index, value)) = self.iter.next() {
            Some((convert_1d_nd_from_shape(index, self.shape), value))
        } else {
            None
        }
    }
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> AsMultiIndex<T, I, NDIM> for I {
    fn multi_index(self, shape: [usize; NDIM]) -> MultiIndexIterator<I, NDIM> {
        MultiIndexIterator::<I, NDIM> { shape, iter: self }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorByValue<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new default iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            pos: 0,
            nelements: arr.len(),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorByRef<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new default iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            pos: 0,
            nelements: arr.len(),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    /// Create a new mutable default iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let nelements = arr.len();
        Self {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorByValue<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { Some(self.arr.imp().get_value_1d_unchecked(self.pos)) };
            self.pos += 1;
            value
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandom1DAccessByRef, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorByRef<'a, ArrayImpl, NDIM>
{
    type Item = &'a ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { Some(self.arr.imp().get_1d_unchecked(self.pos)) };
            self.pos += 1;
            value
        }
    }
}

// In the following have to use transmute to manually change the lifetime of the data
// obtained by `get_mut` to the lifetime 'a of the matrix. The borrow checker cannot see
// that the reference obtained by get_mut is bound to the lifetime of the iterator due
// to the mutable reference in its initialization.
// See also: https://stackoverflow.com/questions/62361624/lifetime-parameter-problem-in-custom-iterator-over-mutable-references
// And also: https://users.rust-lang.org/t/when-is-transmuting-lifetimes-useful/56140

impl<'a, ArrayImpl: UnsafeRandom1DAccessMut, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
{
    type Item = &'a mut ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe {
                std::mem::transmute::<
                    &mut <ArrayImpl as BaseItem>::Item,
                    &'a mut <ArrayImpl as BaseItem>::Item,
                >(self.arr.imp_mut().get_1d_unchecked_mut(self.pos))
            };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Row iterator
pub struct RowIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, ArrayImpl> RowIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, 2>) -> Self {
        let nrows = arr.shape()[0];
        RowIterator {
            arr,
            nrows,
            current_row: 0,
        }
    }
}

/// Mutable row iterator
pub struct RowIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, ArrayImpl> RowIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, 2>) -> Self {
        let nrows = arr.shape()[0];
        RowIteratorMut {
            arr,
            nrows,
            current_row: 0,
        }
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for RowIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<reference::ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r().slice(0, self.current_row);
        self.current_row += 1;
        Some(slice)
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for RowIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r_mut().slice(0, self.current_row);
        self.current_row += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<ArraySlice<ArrayRefMut<'_, ArrayImpl, 2>, 2, 1>, 1>,
                Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

/// Column iterator
pub struct ColIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, ArrayImpl> ColIterator<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a Array<ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        ColIterator {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

/// Mutable column iterator
pub struct ColIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, ArrayImpl> ColIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Create a new column iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, 2>) -> Self {
        let ncols = arr.shape()[1];
        ColIteratorMut {
            arr,
            ncols,
            current_col: 0,
        }
    }
}

impl<'a, ArrayImpl: Shape<2>> std::iter::Iterator for ColIterator<'a, ArrayImpl, 2> {
    type Item = Array<ArraySlice<reference::ArrayRef<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r().slice(1, self.current_col);
        self.current_col += 1;
        Some(slice)
    }
}

impl<'a, ArrayImpl> std::iter::Iterator for ColIteratorMut<'a, ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    type Item = Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r_mut().slice(1, self.current_col);
        self.current_col += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<ArraySlice<ArrayRefMut<'_, ArrayImpl, 2>, 2, 1>, 1>,
                Array<ArraySlice<ArrayRefMut<'a, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

/// Diagonal iterator.
pub struct ArrayDiagIteratorByValue<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorByValue<'a, ArrayImpl, NDIM> {
    /// Create a new diagonal iterator.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        let nelements = *arr.shape().iter().min().unwrap();
        ArrayDiagIteratorByValue {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorByValue<'a, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { self.arr.get_value_unchecked([self.pos; NDIM]) };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Diagonal iterator.
pub struct ArrayDiagIteratorByRef<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorByRef<'a, ArrayImpl, NDIM> {
    /// Create a new diagonal iterator.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        let nelements = *arr.shape().iter().min().unwrap();
        ArrayDiagIteratorByRef {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessByRef<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorByRef<'a, ArrayImpl, NDIM>
{
    type Item = &'a ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { self.arr.get_unchecked([self.pos; NDIM]) };
            self.pos += 1;
            Some(value)
        }
    }
}

/// Mutable diagonal iterator.
pub struct ArrayDiagIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIteratorMut<'a, ArrayImpl, NDIM> {
    /// Create a new mutable diagonal iterator for the given array.
    pub fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let nelements = *arr.shape().iter().min().unwrap();
        ArrayDiagIteratorMut {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessMut<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIteratorMut<'a, ArrayImpl, NDIM>
{
    type Item = &'a mut ArrayImpl::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe {
                std::mem::transmute::<&mut ArrayImpl::Item, Self::Item>(
                    self.arr.get_unchecked_mut([self.pos; NDIM]),
                )
            };
            self.pos += 1;
            Some(value)
        }
    }
}
