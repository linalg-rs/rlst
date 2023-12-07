//! Operations on arrays.
use crate::layout::convert_1d_nd_from_shape;

use super::*;

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + Shape<NDIM>
            + UnsafeRandomAccessMut<NDIM, Item = Item>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Set all elements of an array to zero.
    pub fn set_zero(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::Zero>::zero();
        }
    }

    /// Set all elements of an array to one.
    pub fn set_one(&mut self) {
        for elem in self.iter_mut() {
            *elem = <Item as num::One>::one();
        }
    }

    /// Fill the diagonal of an array with the value 1 and all other elements zero.
    pub fn set_identity(&mut self) {
        self.set_zero();

        for index in 0..self.shape().iter().copied().min().unwrap() {
            *self.get_mut([index; NDIM]).unwrap() = <Item as num::One>::one();
        }
    }

    /// Multiply all array elements with the scalar `alpha`.
    pub fn scale_in_place(&mut self, alpha: Item) {
        for elem in self.iter_mut() {
            *elem *= alpha;
        }
    }

    /// Get the diagonal of an array.
    ///
    /// Argument must be a 1d array of length `self.shape().iter().min()`.
    pub fn get_diag<
        ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1, Item = Item>,
    >(
        &self,
        mut other: Array<Item, ArrayImplOther, 1>,
    ) {
        assert_eq!(
            other.number_of_elements(),
            *self.shape().iter().min().unwrap()
        );
        for index in 0..self.shape().iter().copied().min().unwrap() {
            *other.get_mut([index]).unwrap() = self.get_value([index; NDIM]).unwrap();
        }
    }

    /// Set the diagonal of an array.
    ///
    /// Argument must be a 1d array of length `self.shape().iter().min()`.
    pub fn set_diag<
        ArrayImplOther: UnsafeRandomAccessByValue<1, Item = Item>
            + Shape<1>
            + UnsafeRandomAccessMut<1, Item = Item>,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, 1>,
    ) {
        assert_eq!(
            other.number_of_elements(),
            *self.shape().iter().min().unwrap()
        );
        for index in 0..self.shape().iter().copied().min().unwrap() {
            *self.get_mut([index; NDIM]).unwrap() = other.get_value([index]).unwrap();
        }
    }

    /// Fill an array with values from another array.
    pub fn fill_from<ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>>(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) {
        assert_eq!(self.shape(), other.shape());

        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item = other_item;
        }
    }

    /// Fill an array with values from an other arrays using chunks of size `N`.
    pub fn fill_from_chunked<
        Other: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Other,
    ) {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let Some(chunk) = other.get_chunk(chunk_index) {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_start + data_index,
                        self.shape(),
                    )) = chunk.data[data_index];
                }
            }
            chunk_index += 1;
        }
    }

    /// Sum other array into array.
    pub fn sum_into<ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>>(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) {
        for (item, other_item) in self.iter_mut().zip(other.iter()) {
            *item += other_item;
        }
    }

    /// Chunked summation into array.
    pub fn sum_into_chunked<
        ArrayImplOther: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM> + ChunkedAccess<N, Item = Item>,
        const N: usize,
    >(
        &mut self,
        other: Array<Item, ArrayImplOther, NDIM>,
    ) where
        Self: ChunkedAccess<N, Item = Item>,
    {
        assert_eq!(self.shape(), other.shape());

        let mut chunk_index = 0;

        while let (Some(mut my_chunk), Some(chunk)) =
            (self.get_chunk(chunk_index), other.get_chunk(chunk_index))
        {
            let data_start = chunk.start_index;

            for data_index in 0..chunk.valid_entries {
                my_chunk.data[data_index] += chunk.data[data_index];
            }

            for data_index in 0..chunk.valid_entries {
                unsafe {
                    *self.get_unchecked_mut(convert_1d_nd_from_shape(
                        data_index + data_start,
                        self.shape(),
                    )) = my_chunk.data[data_index];
                }
            }

            chunk_index += 1;
        }
    }
}

impl<
        Item: Scalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item> + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
{
    /// Return true of array is empty (that is one dimension is zero), otherwise false.
    pub fn is_empty(&self) -> bool {
        self.shape().iter().copied().min().unwrap() == 0
    }
}
