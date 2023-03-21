//! Data containers are simple structures that hold data and allow access to it.
//!
//! A data container in `householder` is defined through the [DataContainer] trait.
//! It is a very simple interface that provides low-level access methods and
//! knows about how many elements are in the data container. Some data containers
//! are pre-defined, namely
//!
//!

use crate::types::IndexType;
use crate::types::Scalar;

pub trait DataContainer {
    type Item: Scalar;

    /// Access the container unchecked.
    unsafe fn get_unchecked(&self, index: IndexType) -> Self::Item;

    /// Get pointer to data.
    fn get_pointer(&self) -> *const Self::Item;

    /// Get data slice
    fn get_slice(&self, first: IndexType, last: IndexType) -> &[Self::Item] {
        assert!(
            first < last,
            "'first' {} must be smaller than 'last' {}.",
            first,
            last
        );

        assert!(
            last <= self.number_of_elements(),
            "Value of 'last' {} must be smaller or equal to the number of elements {}.",
            last,
            self.number_of_elements()
        );

        unsafe { std::slice::from_raw_parts(self.get_pointer().add(first), last - first) }
    }

    /// Return the number of elements in the container.
    fn number_of_elements(&self) -> IndexType;
}

pub trait DataContainerMut: DataContainer {
    /// Access the container mutably unchecked.
    unsafe fn get_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item;

    /// Get mutable pointer to data.
    fn get_pointer_mut(&mut self) -> *mut Self::Item;

    /// Get data slice
    fn get_slice_mut(&mut self, first: IndexType, last: IndexType) -> &mut [Self::Item] {
        assert!(
            first < last,
            "'first' {} must be smaller than 'last' {}.",
            first,
            last
        );

        assert!(
            first < self.number_of_elements(),
            "Value of 'first' {} must be smaller than number of elements {}.",
            first,
            self.number_of_elements()
        );

        assert!(
            last <= self.number_of_elements(),
            "Value of 'last' {} must be smaller or equal to the number of elements {}.",
            last,
            self.number_of_elements()
        );

        unsafe { std::slice::from_raw_parts_mut(self.get_pointer_mut().add(first), last - first) }
    }
}

/// A container that uses dynamic vectors.
pub struct VectorContainer<Item: Scalar> {
    data: Vec<Item>,
}

pub struct ArrayContainer<Item: Scalar, const N: usize> {
    data: [Item; N],
}

/// A container that takes a reference to a slice.
pub struct SliceContainer<'a, Item: Scalar> {
    data: &'a [Item],
}

/// A container that takes a mutable reference to a slice.
pub struct SliceContainerMut<'a, Item: Scalar> {
    data: &'a mut [Item],
}

impl<Item: Scalar> VectorContainer<Item> {
    /// New vector container by specifying the number of elements.
    ///
    /// The container is initialized with zeros by default.
    pub fn new(nelems: IndexType) -> VectorContainer<Item> {
        VectorContainer::<Item> {
            data: vec![num::cast::<f64, Item>(0.0).unwrap(); nelems],
        }
    }
}

impl<Item: Scalar, const N: usize> ArrayContainer<Item, N> {
    pub fn new() -> ArrayContainer<Item, N> {
        ArrayContainer::<Item, N> {
            data: [num::cast::<f64, Item>(0.0).unwrap(); N],
        }
    }
}

impl<Item: Scalar, const N: usize> Default for ArrayContainer<Item, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, Item: Scalar> SliceContainer<'a, Item> {
    /// New slice container from a reference.
    pub fn new(slice: &'a [Item]) -> SliceContainer<Item> {
        SliceContainer::<Item> { data: slice }
    }
}

impl<'a, Item: Scalar> SliceContainerMut<'a, Item> {
    /// New mutable slice container from mutable reference.
    pub fn new(slice: &'a mut [Item]) -> SliceContainerMut<Item> {
        SliceContainerMut::<Item> { data: slice }
    }
}

impl<Item: Scalar> DataContainer for VectorContainer<Item> {
    type Item = Item;

    unsafe fn get_unchecked(&self, index: IndexType) -> Self::Item {
        *self.data.get_unchecked(index)
    }

    fn get_pointer(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn number_of_elements(&self) -> IndexType {
        self.data.len()
    }
}

impl<Item: Scalar> DataContainerMut for VectorContainer<Item> {
    unsafe fn get_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item {
        self.data.get_unchecked_mut(index)
    }

    fn get_pointer_mut(&mut self) -> *mut Self::Item {
        self.data.as_mut_ptr()
    }
}

impl<Item: Scalar, const N: usize> DataContainer for ArrayContainer<Item, N> {
    type Item = Item;

    unsafe fn get_unchecked(&self, index: IndexType) -> Self::Item {
        *self.data.get_unchecked(index)
    }

    fn get_pointer(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn number_of_elements(&self) -> IndexType {
        self.data.len()
    }
}

impl<Item: Scalar, const N: usize> DataContainerMut for ArrayContainer<Item, N> {
    unsafe fn get_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item {
        self.data.get_unchecked_mut(index)
    }

    fn get_pointer_mut(&mut self) -> *mut Self::Item {
        self.data.as_mut_ptr()
    }
}

impl<'a, Item: Scalar> DataContainer for SliceContainer<'a, Item> {
    type Item = Item;

    unsafe fn get_unchecked(&self, index: IndexType) -> Self::Item {
        *self.data.get_unchecked(index)
    }

    fn get_pointer(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn number_of_elements(&self) -> IndexType {
        self.data.len()
    }
}

impl<'a, Item: Scalar> DataContainer for SliceContainerMut<'a, Item> {
    type Item = Item;

    unsafe fn get_unchecked(&self, index: IndexType) -> Self::Item {
        *self.data.get_unchecked(index)
    }

    fn get_pointer(&self) -> *const Self::Item {
        self.data.as_ptr()
    }

    fn number_of_elements(&self) -> IndexType {
        self.data.len()
    }
}

impl<'a, Item: Scalar> DataContainerMut for SliceContainerMut<'a, Item> {
    unsafe fn get_unchecked_mut(&mut self, index: IndexType) -> &mut Self::Item {
        self.data.get_unchecked_mut(index)
    }

    fn get_pointer_mut(&mut self) -> *mut Self::Item {
        self.data.as_mut_ptr()
    }
}
