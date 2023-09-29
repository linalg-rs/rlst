//! Various iterator implementations

use crate::array::*;
use crate::layout::convert_1d_nd_from_shape;
use rlst_common::types::Scalar;

pub struct ArrayDefaultIterator<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    shape: [usize; NDIM],
    pos: usize,
    nelements: usize,
}

pub struct ArrayDefaultIteratorMut<
    'a,
    Item: Scalar,
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
        Item: Scalar,
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
        Item: Scalar,
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
        Item: Scalar,
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
        Item: Scalar,
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
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > rlst_common::traits::iterators::DefaultIterator for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type Iter<'a> = ArrayDefaultIterator<'a, Item, ArrayImpl, NDIM> where Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        ArrayDefaultIterator::new(self)
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > rlst_common::traits::iterators::DefaultIteratorMut for Array<Item, ArrayImpl, NDIM>
{
    type Item = Item;
    type IterMut<'a> = ArrayDefaultIteratorMut<'a, Item, ArrayImpl, NDIM> where Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_> {
        ArrayDefaultIteratorMut::new(self)
    }
}

#[cfg(test)]
mod test {

    use rlst_common::traits::*;

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
}
