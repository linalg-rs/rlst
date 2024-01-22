//! Data containers are simple structures that hold data and allow access to it.
//!
//! A data container is defined through the [DataContainer] trait.
//! It is a very simple interface that provides low-level access methods and
//! knows about how many elements are in the data container. Some data containers
//! are pre-defined, namely
//! - [VectorContainer] - A container based on dynamic memory allocation
//! - [ArrayContainer] - A container using compile-time memory allocation
//! - [SliceContainer] - A container that forwards calls to a given memory slice.
//! - [SliceContainerMut] - Like [SliceContainer] but provides also mutable access.
//!

use num::Zero;
use rlst_common::types::RlstScalar;

/// Defines the basic behaviour of a data container.
pub trait DataContainer {
    type Item: RlstScalar;

    /// Access the container unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item;

    /// Access the container by reference
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;

    /// Return the number of elements in the container.
    fn number_of_elements(&self) -> usize;

    /// Get slice to the whole data
    fn data(&self) -> &[Self::Item];
}

pub trait DataContainerMut: DataContainer {
    /// Access the container mutably unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;

    /// Get mutable slice to the whole data.
    fn data_mut(&mut self) -> &mut [Self::Item];
}

/// Definition of a resizeable data container.
///
/// A resizeable data container can change its size during runtime.
pub trait ResizeableDataContainerMut: DataContainerMut {
    /// Resize the data container.
    fn resize(&mut self, new_len: usize);
}

/// A container that uses dynamic vectors.
#[derive(Clone)]
pub struct VectorContainer<Item: RlstScalar> {
    data: Vec<Item>,
}

/// A container that uses a statically allocated array.
///
/// The size of this container needs to be known at compile time.
/// It is useful for data structures that should be stack allocated.
#[derive(Clone)]
pub struct ArrayContainer<Item: RlstScalar, const N: usize> {
    data: [Item; N],
}

/// A container that takes a reference to a slice.
pub struct SliceContainer<'a, Item: RlstScalar> {
    data: &'a [Item],
}

/// A container that takes a mutable reference to a slice.
pub struct SliceContainerMut<'a, Item: RlstScalar> {
    data: &'a mut [Item],
}

impl<Item: RlstScalar> VectorContainer<Item> {
    /// New vector container by specifying the number of elements.
    ///
    /// The container is initialized with zeros by default.
    pub fn new(nelems: usize) -> VectorContainer<Item> {
        VectorContainer::<Item> {
            data: vec![num::cast::<f64, Item>(0.0).unwrap(); nelems],
        }
    }
}

impl<Item: RlstScalar, const N: usize> ArrayContainer<Item, N> {
    pub fn new() -> ArrayContainer<Item, N> {
        ArrayContainer::<Item, N> {
            data: [num::cast::<f64, Item>(0.0).unwrap(); N],
        }
    }
}

impl<Item: RlstScalar, const N: usize> Default for ArrayContainer<Item, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, Item: RlstScalar> SliceContainer<'a, Item> {
    /// New slice container from a reference.
    pub fn new(slice: &'a [Item]) -> SliceContainer<Item> {
        SliceContainer::<Item> { data: slice }
    }
}

impl<'a, Item: RlstScalar> SliceContainerMut<'a, Item> {
    /// New mutable slice container from mutable reference.
    pub fn new(slice: &'a mut [Item]) -> SliceContainerMut<Item> {
        SliceContainerMut::<Item> { data: slice }
    }
}

impl<Item: RlstScalar> DataContainer for VectorContainer<Item> {
    type Item = Item;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }

    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: RlstScalar> DataContainerMut for VectorContainer<Item> {
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked_mut(index)
    }

    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

impl<Item: RlstScalar> ResizeableDataContainerMut for VectorContainer<Item> {
    fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, <Item as Zero>::zero());
    }
}

impl<Item: RlstScalar, const N: usize> DataContainer for ArrayContainer<Item, N> {
    type Item = Item;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }

    fn data(&self) -> &[Self::Item] {
        &self.data
    }

    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: RlstScalar, const N: usize> DataContainerMut for ArrayContainer<Item, N> {
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked_mut(index)
    }

    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

impl<'a, Item: RlstScalar> DataContainer for SliceContainer<'a, Item> {
    type Item = Item;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }

    fn data(&self) -> &[Self::Item] {
        self.data
    }

    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<'a, Item: RlstScalar> DataContainer for SliceContainerMut<'a, Item> {
    type Item = Item;

    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }

    fn data(&self) -> &[Self::Item] {
        self.data
    }

    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<'a, Item: RlstScalar> DataContainerMut for SliceContainerMut<'a, Item> {
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked_mut(index)
    }

    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data
    }
}
