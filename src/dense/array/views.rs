//! Views onto an array.
//!
//! A view onto an array stores a reference to the array and forwards all method calls to the
//! original array. A subview is similar but restricts to a subpart of the original array.

pub mod flattened;
pub mod subview;
pub mod view;

pub use flattened::ArrayFlatView;
pub use subview::ArraySubView;
pub use view::{ArrayView, ArrayViewMut};

use crate::dense::traits::Shape;

use super::Array;

impl<ArrayImpl: Shape<NDIM>, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Move the array into a subview specified by an offset and shape of the subview.
    pub fn into_subview(
        self,
        offset: [usize; NDIM],
        shape: [usize; NDIM],
    ) -> Array<ArraySubView<ArrayImpl, NDIM>, NDIM> {
        Array::new(ArraySubView::new(self, offset, shape))
    }
}

impl<ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM> {
    /// Return a flattened 1d view onto the array. The view is flattened in column-major order.
    pub fn into_flat(self) -> Array<ArrayFlatView<ArrayImpl, NDIM>, 1> {
        Array::new(ArrayFlatView::new(self))
    }
}
