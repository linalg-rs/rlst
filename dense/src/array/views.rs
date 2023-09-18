//! Views onto an array

use super::Array;
use rlst_common::traits::*;

pub struct ArrayView<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
    const NDIM: usize,
> {
    arr: &'a Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

pub struct ArrayViewMut<
    'a,
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
        + Shape<NDIM>
        + UnsafeRandomAccessMut<NDIM, Item = Item>,
    const NDIM: usize,
> {
    arr: &'a mut Array<Item, ArrayImpl, NDIM>,
    offset: [usize; NDIM],
    shape: [usize; NDIM],
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > ArrayView<'a, Item, ArrayImpl, NDIM>
{
    pub fn new(
        arr: &'a Array<Item, ArrayImpl, NDIM>,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Self {
        let arr_shape = arr.shape();
        for index in 0..NDIM {
            assert!(
                offset[index] + shape[index] <= arr_shape[index],
                "View out of bounds for dimension {}. {} > {}",
                index,
                offset[index] + shape[index],
                arr_shape[index]
            )
        }
        Self { arr, offset, shape }
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    pub fn new(
        arr: &'a mut Array<Item, ArrayImpl, NDIM>,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Self {
        let arr_shape = arr.shape();
        for index in 0..NDIM {
            assert!(
                offset[index] + shape[index] <= arr_shape[index],
                "View out of bounds for dimension {}. {} > {}",
                index,
                offset[index] + shape[index],
                arr_shape[index]
            )
        }
        Self { arr, offset, shape }
    }
}

// Basic traits for ArrayView

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.arr
            .get_value_unchecked(offset_indices(indices, self.offset))
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayView<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, indices: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(offset_indices(indices, self.offset))
    }
}

// Basic traits for ArrayViewMut

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Shape<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    fn shape(&self) -> [usize; NDIM] {
        self.shape
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByValue<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_value_unchecked(&self, indices: [usize; NDIM]) -> Self::Item {
        self.arr
            .get_value_unchecked(offset_indices(indices, self.offset))
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessByRef<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessByRef<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked(&self, indices: [usize; NDIM]) -> &Self::Item {
        self.arr.get_unchecked(offset_indices(indices, self.offset))
    }
}

impl<
        'a,
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > UnsafeRandomAccessMut<NDIM> for ArrayViewMut<'a, Item, ArrayImpl, NDIM>
{
    type Item = Item;
    #[inline]
    unsafe fn get_unchecked_mut(&mut self, indices: [usize; NDIM]) -> &mut Self::Item {
        self.arr
            .get_unchecked_mut(offset_indices(indices, self.offset))
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn subview(
        &self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayView::new(self, offset, shape))
    }

    pub fn view(&self) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        self.subview([0; NDIM], self.shape())
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    pub fn subview_mut(
        &mut self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayViewMut::new(self, offset, shape))
    }

    pub fn view_mut(&mut self) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        self.subview_mut([0; NDIM], self.shape())
    }
}

fn offset_indices<const NDIM: usize>(
    indices: [usize; NDIM],
    offset: [usize; NDIM],
) -> [usize; NDIM] {
    let mut output = [0; NDIM];
    for (ind, elem) in output.iter_mut().enumerate() {
        *elem = indices[ind] + offset[ind]
    }
    output
}
