//! Various iterator implementations.

use crate::array::*;
use crate::layout::convert_1d_nd_from_shape;
use crate::types::RlstScalar;

use super::slice::ArraySlice;

/// Default column major iterator.
///
/// This iterator returns elements of an array in standard column major order.
pub struct ArrayDefaultIterator<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    shape: [usize; NDIM],
    pos: usize,
    nelements: usize,
}

/// Mutable default iterator. Like [ArrayDefaultIterator] but with mutable access.
pub struct ArrayDefaultIteratorMut<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    shape: [usize; NDIM],
    pos: usize,
    nelements: usize,
}

/// A multi-index iterator returns the corrent element and the corresponding multi-index.
pub struct MultiIndexIterator<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> {
    shape: [usize; NDIM],
    iter: I,
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> Iterator
    for MultiIndexIterator<T, I, NDIM>
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

/// Helper trait that returns from an enumeration iterator a new iterator
/// that converts the 1d index into a multi-index.
pub trait AsMultiIndex<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> {
    fn multi_index(self, shape: [usize; NDIM]) -> MultiIndexIterator<T, I, NDIM>;
}

impl<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> AsMultiIndex<T, I, NDIM> for I {
    fn multi_index(self, shape: [usize; NDIM]) -> MultiIndexIterator<T, I, NDIM> {
        MultiIndexIterator::<T, I, NDIM> { shape, iter: self }
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayDefaultIterator<'a, Item, ArrayImpl, NDIM>
{
    fn new(arr: &'a Array<Item, ArrayImpl, NDIM>) -> Self {
        Self {
            arr,
            shape: arr.shape(),
            pos: 0,
            nelements: arr.shape().iter().product(),
        }
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > ArrayDefaultIteratorMut<'a, Item, ArrayImpl, NDIM>
{
    fn new(arr: &'a mut Array<Item, ArrayImpl, NDIM>) -> Self {
        let shape = arr.shape();
        Self {
            arr,
            shape,
            pos: 0,
            nelements: shape.iter().product(),
        }
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > std::iter::Iterator for ArrayDefaultIterator<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            return None;
        }
        let multi_index = convert_1d_nd_from_shape(self.pos, self.shape);
        self.pos += 1;
        unsafe { Some(self.arr.get_value_unchecked(multi_index)) }
    }
}

// In the following have to use transmute to manually change the lifetime of the data
// obtained by `get_mut` to the lifetime 'a of the matrix. The borrow checker cannot see
// that the reference obtained by get_mut is bound to the lifetime of the iterator due
// to the mutable reference in its initialization.
// See also: https://stackoverflow.com/questions/62361624/lifetime-parameter-problem-in-custom-iterator-over-mutable-references
// And also: https://users.rust-lang.org/t/when-is-transmuting-lifetimes-useful/56140

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + Shape<NDIM>,
        const NDIM: usize,
    > std::iter::Iterator for ArrayDefaultIteratorMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = &'a mut Item;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.nelements {
            return None;
        }
        let multi_index = convert_1d_nd_from_shape(self.pos, self.shape);
        self.pos += 1;
        unsafe {
            Some(std::mem::transmute::<&mut Item, &'a mut Item>(
                self.arr.get_unchecked_mut(multi_index),
            ))
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > crate::traits::DefaultIterator for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type Iter<'a> = ArrayDefaultIterator<'a, Item, ArrayImpl, NDIM> where Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        ArrayDefaultIterator::new(self)
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > crate::traits::DefaultIteratorMut for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type IterMut<'a> = ArrayDefaultIteratorMut<'a, Item, ArrayImpl, NDIM> where Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        ArrayDefaultIteratorMut::new(self)
    }
}

pub struct RowIterator<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

pub struct RowIteratorMut<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    nrows: usize,
    current_row: usize,
}

impl<'a, Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    std::iter::Iterator for RowIterator<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, views::ArrayView<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.view().slice(0, self.current_row);
        self.current_row += 1;
        Some(slice)
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>,
    > std::iter::Iterator for RowIteratorMut<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, views::ArrayViewMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.nrows {
            return None;
        }
        let slice = self.arr.view_mut().slice(0, self.current_row);
        self.current_row += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<Item, ArraySlice<Item, views::ArrayViewMut<'_, Item, ArrayImpl, 2>, 2, 1>, 1>,
                Array<Item, ArraySlice<Item, views::ArrayViewMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    Array<Item, ArrayImpl, 2>
{
    /// Return a row iterator for a two-dimensional array.
    pub fn row_iter(&self) -> RowIterator<'_, Item, ArrayImpl, 2> {
        RowIterator {
            arr: self,
            nrows: self.shape()[0],
            current_row: 0,
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
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

pub struct ColIterator<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

pub struct ColIteratorMut<
    'a,
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    ncols: usize,
    current_col: usize,
}

impl<'a, Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    std::iter::Iterator for ColIterator<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, views::ArrayView<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.view().slice(1, self.current_col);
        self.current_col += 1;
        Some(slice)
    }
}

impl<
        'a,
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Shape<2>,
    > std::iter::Iterator for ColIteratorMut<'a, Item, ArrayImpl, 2>
{
    type Item = Array<Item, ArraySlice<Item, views::ArrayViewMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_col >= self.ncols {
            return None;
        }
        let slice = self.arr.view_mut().slice(1, self.current_col);
        self.current_col += 1;
        unsafe {
            Some(std::mem::transmute::<
                Array<Item, ArraySlice<Item, views::ArrayViewMut<'_, Item, ArrayImpl, 2>, 2, 1>, 1>,
                Array<Item, ArraySlice<Item, views::ArrayViewMut<'a, Item, ArrayImpl, 2>, 2, 1>, 1>,
            >(slice))
        }
    }
}

impl<Item: RlstScalar, ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2>>
    Array<Item, ArrayImpl, 2>
{
    /// Return a column iterator for a two-dimensional array.
    pub fn col_iter(&self) -> ColIterator<'_, Item, ArrayImpl, 2> {
        ColIterator {
            arr: self,
            ncols: self.shape()[1],
            current_col: 0,
        }
    }
}

impl<
        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
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

#[cfg(test)]
mod test {

    use approx::assert_relative_eq;

    use crate::traits::*;

    #[test]
    fn test_iter() {
        let mut arr = crate::rlst_dynamic_array3![f64, [1, 3, 2]];

        for (index, item) in arr.iter_mut().enumerate() {
            *item = index as f64;
        }

        assert_eq!(arr[[0, 0, 0]], 0.0);
        assert_eq!(arr[[0, 1, 0]], 1.0);
        assert_eq!(arr[[0, 2, 0]], 2.0);
        assert_eq!(arr[[0, 0, 1]], 3.0);
        assert_eq!(arr[[0, 1, 1]], 4.0);
        assert_eq!(arr[[0, 2, 1]], 5.0);
    }

    #[test]
    fn test_row_iter() {
        let shape = [2, 3];
        let mut arr = crate::rlst_dynamic_array2![f64, shape];

        arr.fill_from_seed_equally_distributed(0);

        let mut row_count = 0;
        for (row_index, row) in arr.row_iter().enumerate() {
            for col_index in 0..shape[1] {
                assert_eq!(row[[col_index]], arr[[row_index, col_index]]);
            }
            row_count += 1;
        }
        assert_eq!(row_count, shape[0]);
    }

    #[test]
    fn test_row_iter_mut() {
        let shape = [2, 3];
        let mut arr = crate::rlst_dynamic_array2![f64, shape];
        let mut arr2 = crate::rlst_dynamic_array2![f64, shape];

        arr.fill_from_seed_equally_distributed(0);
        arr2.fill_from(arr.view());

        let mut row_count = 0;
        for (row_index, mut row) in arr.row_iter_mut().enumerate() {
            for col_index in 0..shape[1] {
                row[[col_index]] *= 2.0;
                assert_relative_eq!(
                    row[[col_index]],
                    2.0 * arr2[[row_index, col_index]],
                    epsilon = 1E-14
                );
            }
            row_count += 1;
        }
        assert_eq!(row_count, shape[0]);
    }

    #[test]
    fn test_col_iter() {
        let shape = [2, 3];
        let mut arr = crate::rlst_dynamic_array2![f64, shape];

        arr.fill_from_seed_equally_distributed(0);

        let mut col_count = 0;
        for (col_index, col) in arr.col_iter().enumerate() {
            for row_index in 0..shape[0] {
                assert_eq!(col[[row_index]], arr[[row_index, col_index]]);
            }
            col_count += 1;
        }

        assert_eq!(col_count, shape[1]);
    }

    #[test]
    fn test_col_iter_mut() {
        let shape = [2, 3];
        let mut arr = crate::rlst_dynamic_array2![f64, shape];
        let mut arr2 = crate::rlst_dynamic_array2![f64, shape];

        arr.fill_from_seed_equally_distributed(0);
        arr2.fill_from(arr.view());

        let mut col_count = 0;
        for (col_index, mut col) in arr.col_iter_mut().enumerate() {
            for row_index in 0..shape[0] {
                col[[row_index]] *= 2.0;
                assert_relative_eq!(
                    col[[row_index]],
                    2.0 * arr2[[row_index, col_index]],
                    epsilon = 1E-14
                );
            }
            col_count += 1;
        }
        assert_eq!(col_count, shape[1]);
    }
}
