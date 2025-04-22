//! Views onto an array.
//!
//! A view onto an array stores a reference to the array and forwards all method calls to the
//! original array. A subview is similar but restricts to a subpart of the original array.

pub mod flattened;
pub mod subview;
pub mod view;

pub use flattened::{ArrayFlatView, ArrayFlatViewMut};
pub use subview::ArraySubView;
pub use view::{ArrayView, ArrayViewMut};

use crate::{
    dense::{
        layout::convert_nd_raw,
        traits::{UnsafeRandom1DAccessByValue, UnsafeRandom1DAccessMut},
        types::RlstBase,
    },
    Array, Shape, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};

// Basic traits for ArrayViewMut

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Move the array into a subview specified by an offset and shape of the subview.
    pub fn into_subview(
        self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<Item, ArraySubView<Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArraySubView::new(self, offset, shape))
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandom1DAccessByValue<Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return a view onto the array.
    #[deprecated(note = "Please use arr.r() instead.")]
    pub fn view(&self) -> Array<Item, ArrayView<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayView::new(self))
    }

    /// Return a flattened 1d view onto the array. The view is flattened in column-major order.
    pub fn view_flat(&self) -> Array<Item, ArrayFlatView<'_, Item, ArrayImpl, NDIM>, 1> {
        Array::new(ArrayFlatView::new(self))
    }
}

impl<
        Item: RlstBase,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + UnsafeRandom1DAccessByValue<Item = Item>
            + UnsafeRandom1DAccessMut<Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return a mutable view onto the array.
    #[deprecated(note = "Please use arr.r_mut() instead.")]
    pub fn view_mut(&mut self) -> Array<Item, ArrayViewMut<'_, Item, ArrayImpl, NDIM>, NDIM> {
        Array::new(ArrayViewMut::new(self))
    }

    /// Return a flattened 1d view onto the array. The view is flattened in column-major order.
    pub fn view_flat_mut(&mut self) -> Array<Item, ArrayFlatViewMut<'_, Item, ArrayImpl, NDIM>, 1> {
        Array::new(ArrayFlatViewMut::new(self))
    }
}

fn offset_multi_index<const NDIM: usize>(
    multi_index: [usize; NDIM],
    offset: [usize; NDIM],
) -> [usize; NDIM] {
    let mut output = [0; NDIM];
    for (ind, elem) in output.iter_mut().enumerate() {
        *elem = multi_index[ind] + offset[ind]
    }
    output
}

fn compute_raw_range<const NDIM: usize>(
    offset: [usize; NDIM],
    stride: [usize; NDIM],
    shape: [usize; NDIM],
) -> (usize, usize) {
    let start_multi_index = offset;
    let mut end_multi_index = [0; NDIM];
    for (index, value) in end_multi_index.iter_mut().enumerate() {
        let sum = start_multi_index[index] + shape[index];
        assert!(sum > 0);
        *value = sum - 1;
    }

    let start_raw = convert_nd_raw(start_multi_index, stride);
    let end_raw = convert_nd_raw(end_multi_index, stride);
    (start_raw, 1 + end_raw)
}
