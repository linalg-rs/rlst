//! This module contains methods only defined for matrices of type
//! [GenericBaseMatrix](crate::matrix::GenericBaseMatrix) or
//! [GenericBaseMatrixMut].

use crate::data_container::{DataContainer, DataContainerMut};
use crate::traits::*;
use crate::types::*;

use super::{GenericBaseMatrix, GenericBaseMatrixMut};

impl<
        Item: Scalar,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = Item>,
    > GenericBaseMatrixMut<Item, L, Data, RS, CS>
{
    /// Apply a callable to each element of a matrix.
    ///
    /// The callable `f` takes a mutable reference to a matrix
    /// element.
    pub fn for_each<F: FnMut(&mut Item)>(&mut self, mut f: F) {
        for index in 0..self.layout().number_of_elements() {
            unsafe { f(self.get1d_unchecked_mut(index)) }
        }
    }
}

impl<
        Item: Scalar,
        Data: DataContainer<Item = Item>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > GenericBaseMatrix<Item, L, Data, RS, CS>
{
    /// Return a pointer to the start of the underlying memory region.
    #[inline]
    pub fn get_pointer(&self) -> *const Item {
        self.0.get_pointer()
    }

    /// Return a region of the matrix as memory slice.
    ///
    /// The parameters `first` and `last` are with respect to raw
    /// indexing in the memory. So for nontrivial strides this will
    /// be different to the 1d indices in standard row or column major
    /// order. To convert from a simple 1d index to a raw index use the method
    /// [crate::traits::LayoutType::convert_1d_raw] or to convert from a 2d index
    /// to a raw index use [crate::traits::LayoutType::convert_2d_raw].
    #[inline]
    pub fn get_slice(&self, first: IndexType, last: IndexType) -> &[Item] {
        self.0.get_slice(first, last)
    }
}

impl<
        Item: Scalar,
        Data: DataContainerMut<Item = Item>,
        L: LayoutType,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
    > GenericBaseMatrixMut<Item, L, Data, RS, CS>
{
    /// Return a pointer to the start of the underlying memory region.
    #[inline]
    pub fn get_pointer_mut(&mut self) -> *mut Item {
        self.0.get_pointer_mut()
    }

    /// Return a region of the matrix as mutable memory slice.
    ///
    /// The parameters `first` and `last` are with respect to raw
    /// indexing in the memory. So for nontrivial strides this will
    /// be different to the 1d indices in standard row or column major
    /// order. To convert from a simple 1d index to a raw index use the method
    /// [crate::traits::LayoutType::convert_1d_raw] or to convert from a 2d index
    /// to a raw index use [crate::traits::LayoutType::convert_2d_raw].
    #[inline]
    pub fn get_slice_mut(&mut self, first: IndexType, last: IndexType) -> &mut [Item] {
        self.0.get_slice_mut(first, last)
    }
}
