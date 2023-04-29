//! This module implements traits specific to
//! [GenericBaseMatrix](crate::matrix::GenericBaseMatrix)

use crate::data_container::{DataContainer, DataContainerMut};
use crate::traits::*;
use rlst_common::traits::*;

use super::GenericBaseMatrix;

impl<Item: Scalar, RS: SizeIdentifier, CS: SizeIdentifier, Data: DataContainerMut<Item = Item>>
    ForEach for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;
    fn for_each<F: FnMut(&mut Self::T)>(&mut self, mut f: F) {
        for index in 0..self.layout().number_of_elements() {
            unsafe { f(self.get1d_unchecked_mut(index)) }
        }
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    RawAccess for GenericBaseMatrix<Item, Data, RS, CS>
{
    type T = Item;

    #[inline]
    fn get_pointer(&self) -> *const Item {
        self.0.get_pointer()
    }

    #[inline]
    fn get_slice(&self, first: usize, last: usize) -> &[Item] {
        self.0.get_slice(first, last)
    }

    #[inline]
    fn data(&self) -> &[Item] {
        self.0.get_slice(0, self.layout().number_of_elements())
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, RS: SizeIdentifier, CS: SizeIdentifier>
    RawAccessMut for GenericBaseMatrix<Item, Data, RS, CS>
{
    fn get_pointer_mut(&mut self) -> *mut Item {
        self.0.get_pointer_mut()
    }

    fn get_slice_mut(&mut self, first: usize, last: usize) -> &mut [Item] {
        self.0.get_slice_mut(first, last)
    }

    fn data_mut(&mut self) -> &mut [Item] {
        self.0.get_slice_mut(0, self.layout().number_of_elements())
    }
}
