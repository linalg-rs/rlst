//! Definition of a base matrix.
//!
//! A base matrix is an implementation of a matrix that
//! is directly associated with a [DataContainer] (typically a memory region).
//!
//! The user should never interact with [BaseMatrix] directly. Rather, the
//! relevant user type is a [GenericBaseMatrix](crate::matrix::GenericBaseMatrix),
//! which is a [Matrix](crate::matrix::Matrix) that forwards call to the
//! [BaseMatrix] implementation.
//!
use crate::data_container::{DataContainer, DataContainerMut};
use crate::types::Scalar;
use crate::{traits::*, DefaultLayout};
use std::marker::PhantomData;

pub struct BaseMatrix<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> {
    data: Data,
    layout: DefaultLayout,
    phantom_s: PhantomData<S>,
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> BaseMatrix<Item, Data, S> {
    pub fn new(data: Data, layout: DefaultLayout) -> Self {
        assert!(
            layout.number_of_elements() <= data.number_of_elements(),
            "Number of elements in data: {}. But layout number of elements is {})",
            data.number_of_elements(),
            layout.number_of_elements(),
        );
        BaseMatrix::<Item, Data, S> {
            data,
            layout,
            phantom_s: PhantomData,
        }
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> RawAccess
    for BaseMatrix<Item, Data, S>
{
    type T = Item;

    #[inline]
    fn get_pointer(&self) -> *const Item {
        self.data.get_pointer()
    }

    #[inline]
    fn get_slice(&self, first: usize, last: usize) -> &[Item] {
        self.data.get_slice(first, last)
    }

    #[inline]
    fn data(&self) -> &[Item] {
        self.data.data()
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, S: SizeIdentifier> RawAccessMut
    for BaseMatrix<Item, Data, S>
{
    fn get_pointer_mut(&mut self) -> *mut Item {
        self.data.get_pointer_mut()
    }

    fn get_slice_mut(&mut self, first: usize, last: usize) -> &mut [Item] {
        self.data.get_slice_mut(first, last)
    }

    fn data_mut(&mut self) -> &mut [Item] {
        self.data.data_mut()
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> Layout
    for BaseMatrix<Item, Data, S>
{
    type Impl = DefaultLayout;

    #[inline]
    fn layout(&self) -> &Self::Impl {
        &self.layout
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> MatrixImplIdentifier
    for BaseMatrix<Item, Data, S>
{
    const MAT_IMPL: MatrixImplType = MatrixImplType::Base;
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> Size
    for BaseMatrix<Item, Data, S>
{
    type S = S;
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> UnsafeRandomAccessByValue
    for BaseMatrix<Item, Data, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_value_unchecked(&self, row: usize, col: usize) -> Self::Item {
        self.data
            .get_unchecked_value(self.layout.convert_2d_raw(row, col))
    }

    #[inline]
    unsafe fn get1d_value_unchecked(&self, index: usize) -> Self::Item {
        self.data
            .get_unchecked_value(self.layout.convert_1d_raw(index))
    }
}

impl<Item: Scalar, Data: DataContainer<Item = Item>, S: SizeIdentifier> UnsafeRandomAccessByRef
    for BaseMatrix<Item, Data, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Item {
        self.data
            .get_unchecked(self.layout.convert_2d_raw(row, col))
    }

    #[inline]
    unsafe fn get1d_unchecked(&self, index: usize) -> &Self::Item {
        self.data.get_unchecked(self.layout.convert_1d_raw(index))
    }
}

impl<Item: Scalar, Data: DataContainerMut<Item = Item>, S: SizeIdentifier> UnsafeRandomAccessMut
    for BaseMatrix<Item, Data, S>
{
    type Item = Item;

    #[inline]
    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Item {
        self.data
            .get_unchecked_mut(self.layout.convert_2d_raw(row, col))
    }

    #[inline]
    unsafe fn get1d_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        self.data
            .get_unchecked_mut(self.layout.convert_1d_raw(index))
    }
}
