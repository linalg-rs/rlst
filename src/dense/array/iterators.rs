//! Various iterator implementations.

use crate::dense::array::{Array, Shape};
use crate::dense::layout::convert_1d_nd_from_shape;
use crate::dense::traits::{
    AsMultiIndex, Len, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::BaseItem;

use super::reference::{self, ArrayRefMut};
use super::slice::ArraySlice;

/// Default column major iterator.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIterator<'a, ArrayImpl, const NDIM: usize> {
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

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIterator<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            pos: 0,
            nelements: Len::len(arr),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let nelements = Len::len(arr);
        Self {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<ArrayImpl: UnsafeRandom1DAccessByValue, const NDIM: usize> std::iter::Iterator
    for ArrayDefaultIterator<'_, ArrayImpl, NDIM>
{
    type Item = ArrayImpl::Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            None
        } else {
            let value = unsafe { Some(self.arr.get_value_1d_unchecked(self.pos)) };
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
                >(self.arr.get_1d_unchecked_mut(self.pos))
            };
            self.pos += 1;
            Some(value)
        }
    }
}

impl<ArrayImpl, const NDIM: usize> crate::dense::traits::ArrayIterator for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessByValue + Shape<NDIM>,
{
    type Iter<'a>
        = ArrayDefaultIterator<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        ArrayDefaultIterator::new(self)
    }
}

impl<ArrayImpl, const NDIM: usize> crate::dense::traits::ArrayIteratorMut for Array<ArrayImpl, NDIM>
where
    ArrayImpl: UnsafeRandom1DAccessMut + Shape<NDIM>,
{
    type IterMut<'a>
        = ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        ArrayDefaultIteratorMut::new(self)
    }
}

/// Row iterator
pub struct RowIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

/// Mutable row iterator
pub struct RowIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
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

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Return a row iterator for a two-dimensional array.
    pub fn row_iter(&self) -> RowIterator<'_, ArrayImpl, 2> {
        RowIterator {
            arr: self,
            nrows: self.shape()[0],
            current_row: 0,
        }
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Return a mutable row iterator for a two-dimensional array.
    pub fn row_iter_mut(&mut self) -> RowIteratorMut<'_, ArrayImpl, 2> {
        let nrows = self.shape()[0];
        RowIteratorMut {
            arr: self,
            nrows,
            current_row: 0,
        }
    }
}

/// Column iterator
pub struct ColIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

/// Mutable column iterator
pub struct ColIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
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

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Return a column iterator for a two-dimensional array.
    pub fn col_iter(&self) -> ColIterator<'_, ArrayImpl, 2> {
        ColIterator {
            arr: self,
            ncols: self.shape()[1],
            current_col: 0,
        }
    }
}

impl<ArrayImpl> Array<ArrayImpl, 2>
where
    ArrayImpl: Shape<2>,
{
    /// Return a mutable column iterator for a two-dimensional array.
    pub fn col_iter_mut(&mut self) -> ColIteratorMut<'_, ArrayImpl, 2> {
        let ncols = self.shape()[1];
        ColIteratorMut {
            arr: self,
            ncols,
            current_col: 0,
        }
    }
}

/// Diagonal iterator.
pub struct ArrayDiagIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    pos: usize,
    nelements: usize,
}

impl<'a, ArrayImpl: Shape<NDIM>, const NDIM: usize> ArrayDiagIterator<'a, ArrayImpl, NDIM> {
    /// Create a new diagonal iterator.
    pub fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        let nelements = *arr.shape().iter().min().unwrap();
        ArrayDiagIterator {
            arr,
            pos: 0,
            nelements,
        }
    }
}

impl<'a, ArrayImpl: UnsafeRandomAccessByValue<NDIM>, const NDIM: usize> Iterator
    for ArrayDiagIterator<'a, ArrayImpl, NDIM>
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
