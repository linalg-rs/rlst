//! Various iterator implementations.

use crate::dense::array::{Array, Shape};
use crate::dense::layout::convert_1d_nd_from_shape;
use crate::dense::traits::{
    AsMultiIndex, MutableArrayImpl, UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut,
    ValueArrayImpl,
};
use crate::dense::types::RlstBase;

use super::reference::{self, ArrayRefMut};
use super::slice::ArraySlice;

/// Default column major iterator.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIterator<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a Array<ArrayImpl, NDIM>,
    shape: [usize; NDIM],
    pos: usize,
    nelements: usize,
}

/// Mutable default iterator. Like [ArrayDefaultIterator] but with mutable access.
pub struct ArrayDefaultIteratorMut<'a, ArrayImpl, const NDIM: usize> {
    arr: &'a mut Array<ArrayImpl, NDIM>,
    shape: [usize; NDIM],
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
        MultiIndexIterator::<T, I, NDIM> { shape, iter: self }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIterator<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    fn new(arr: &'a Array<ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            shape: arr.shape(),
            pos: 0,
            nelements: arr.shape().iter().product(),
        }
    }
}

impl<'a, ArrayImpl, const NDIM: usize> ArrayDefaultIteratorMut<'a, ArrayImpl, NDIM>
where
    ArrayImpl: Shape<NDIM>,
{
    fn new(arr: &'a mut Array<ArrayImpl, NDIM>) -> Self {
        let shape = arr.shape();
        Self {
            arr,
            shape,
            pos: 0,
            nelements: shape.iter().product(),
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
                Some(std::mem::transmute::<&mut Self::Item, &'a mut Self::Item>(
                    self.arr.get_1d_unchecked_mut(self.pos),
                ))
            };
            self.pos += 1;
            value
        }
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
    crate::dense::traits::DefaultIterator for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type Iter<'a>
        = ArrayDefaultIterator<'a, Item, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        ArrayDefaultIterator::new(self)
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<NDIM, Item>, const NDIM: usize>
    crate::dense::traits::DefaultIteratorMut for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type IterMut<'a>
        = ArrayDefaultIteratorMut<'a, Item, ArrayImpl, NDIM>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        ArrayDefaultIteratorMut::new(self)
    }
}

/// Row iterator
pub struct RowIterator<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
{
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

/// Mutable row iterator
pub struct RowIteratorMut<
    'a,
    Item: RlstBase,
    ArrayImpl: MutableArrayImpl<NDIM, Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<2, Item>> std::iter::Iterator
    for RowIterator<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, reference::ArrayRef<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r().slice(0, self.current_row);
        self.current_row += 1;
        Some(slice)
    }
}

impl<'a, Item: RlstBase, ArrayImpl: MutableArrayImpl<2, Item>> std::iter::Iterator
    for RowIteratorMut<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, ArrayRefMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.r_mut().slice(0, self.current_row);
        self.current_row += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<Item, ArraySlice<Item, ArrayRefMut<'_, Item, ArrayImpl, 2>, 2, 1>, 1>,
                Array<Item, ArraySlice<Item, ArrayRefMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<2, Item>> Array<Item, ArrayImpl, 2> {
    /// Return a row iterator for a two-dimensional array.
    pub fn row_iter(&self) -> RowIterator<'_, Item, ArrayImpl, 2> {
        RowIterator {
            arr: self,
            nrows: self.shape()[0],
            current_row: 0,
        }
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<2, Item>> Array<Item, ArrayImpl, 2> {
    /// Return a mutable row iterator for a two-dimensional array.
    pub fn row_iter_mut(&mut self) -> RowIteratorMut<'_, Item, ArrayImpl, 2> {
        let nrows = self.shape()[0];
        RowIteratorMut {
            arr: self,
            nrows,
            current_row: 0,
        }
    }
}

/// Column iterator
pub struct ColIterator<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<NDIM, Item>, const NDIM: usize>
{
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

/// Mutable column iterator
pub struct ColIteratorMut<
    'a,
    Item: RlstBase,
    ArrayImpl: MutableArrayImpl<NDIM, Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, Item: RlstBase, ArrayImpl: ValueArrayImpl<2, Item>> std::iter::Iterator
    for ColIterator<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, reference::ArrayRef<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r().slice(1, self.current_col);
        self.current_col += 1;
        Some(slice)
    }
}

impl<'a, Item: RlstBase, ArrayImpl: MutableArrayImpl<2, Item>> std::iter::Iterator
    for ColIteratorMut<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, ArrayRefMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.r_mut().slice(1, self.current_col);
        self.current_col += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<Item, ArraySlice<Item, ArrayRefMut<'_, Item, ArrayImpl, 2>, 2, 1>, 1>,
                Array<Item, ArraySlice<Item, ArrayRefMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

impl<Item: RlstBase, ArrayImpl: ValueArrayImpl<2, Item>> Array<Item, ArrayImpl, 2> {
    /// Return a column iterator for a two-dimensional array.
    pub fn col_iter(&self) -> ColIterator<'_, Item, ArrayImpl, 2> {
        ColIterator {
            arr: self,
            ncols: self.shape()[1],
            current_col: 0,
        }
    }
}

impl<Item: RlstBase, ArrayImpl: MutableArrayImpl<2, Item>> Array<Item, ArrayImpl, 2> {
    /// Return a mutable column iterator for a two-dimensional array.
    pub fn col_iter_mut(&mut self) -> ColIteratorMut<'_, Item, ArrayImpl, 2> {
        let ncols = self.shape()[1];
        ColIteratorMut {
            arr: self,
            ncols,
            current_col: 0,
        }
    }
}
