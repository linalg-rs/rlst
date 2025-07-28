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

use crate::traits::data_container::{
    ContainerTypeHint, DataContainer, ModifiableDataContainer, MutableRawAccessDataContainer,
    RawAccessDataContainer, RefDataContainer, RefDataContainerMut, ResizeableDataContainer,
    ValueDataContainer,
};

use crate::base_types::{Heap, Stack};

/// A container that uses dynamic vectors.
pub struct VectorContainer<Item> {
    data: Vec<Item>,
}

impl<Item: Default + Clone> VectorContainer<Item> {
    /// New vector container by specifying the number of elements.
    ///
    /// The container is initialized with zeros by default.
    #[inline(always)]
    pub fn new(nelems: usize) -> VectorContainer<Item> {
        VectorContainer::<Item> {
            data: vec![<Item as Default>::default(); nelems],
        }
    }

    #[inline(always)]
    /// Create a data container by taking ownership of a vector.
    pub fn from_vec(data: Vec<Item>) -> VectorContainer<Item> {
        VectorContainer::<Item> { data }
    }
}

impl<Item> ContainerTypeHint for VectorContainer<Item> {
    type TypeHint = Heap;
}

impl<Item> DataContainer for VectorContainer<Item> {
    type Item = Item;

    #[inline(always)]
    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: Copy> ValueDataContainer for VectorContainer<Item> {
    #[inline(always)]
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }
}

impl<Item> RefDataContainer for VectorContainer<Item> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }
}

impl<Item> RefDataContainerMut for VectorContainer<Item> {
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked_mut(index)
    }
}

impl<Item> ModifiableDataContainer for VectorContainer<Item> {
    #[inline(always)]
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item) {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked_mut(index) = value;
    }
}

impl<Item: Default> ResizeableDataContainer for VectorContainer<Item> {
    #[inline(always)]
    fn resize(&mut self, new_len: usize) {
        self.data.resize_with(new_len, Default::default);
    }
}

impl<Item> RawAccessDataContainer for VectorContainer<Item> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.data.as_slice()
    }
}

impl<Item> MutableRawAccessDataContainer for VectorContainer<Item> {
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data.as_mut_slice()
    }
}

/// A container that uses a statically allocated array.
///
/// The size of this container needs to be known at compile time.
/// It is useful for data structures that should be stack allocated.
pub struct ArrayContainer<Item, const N: usize> {
    data: [Item; N],
}

impl<Item, const N: usize> ContainerTypeHint for ArrayContainer<Item, N> {
    type TypeHint = Stack<N>;
}

impl<Item: Default + Copy, const N: usize> ArrayContainer<Item, N> {
    /// New array container
    #[inline(always)]
    pub fn new() -> ArrayContainer<Item, N> {
        ArrayContainer::<Item, N> {
            data: [<Item as Default>::default(); N],
        }
    }
}

impl<Item: Default + Copy, const N: usize> Default for ArrayContainer<Item, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<Item: Copy, const N: usize> DataContainer for ArrayContainer<Item, N> {
    type Item = Item;

    #[inline(always)]
    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: Copy, const N: usize> ValueDataContainer for ArrayContainer<Item, N> {
    #[inline(always)]
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < N);
        *self.data.get_unchecked(index)
    }
}

impl<Item: Copy, const N: usize> RefDataContainer for ArrayContainer<Item, N> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < N);
        self.data.get_unchecked(index)
    }
}

impl<Item: Copy, const N: usize> RefDataContainerMut for ArrayContainer<Item, N> {
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < N);
        self.data.get_unchecked_mut(index)
    }
}

impl<Item: Copy, const N: usize> ModifiableDataContainer for ArrayContainer<Item, N> {
    #[inline(always)]
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item) {
        debug_assert!(index < N);
        *self.get_unchecked_mut(index) = value;
    }
}

impl<Item: Copy, const N: usize> RawAccessDataContainer for ArrayContainer<Item, N> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        &self.data
    }
}

impl<Item: Copy, const N: usize> MutableRawAccessDataContainer for ArrayContainer<Item, N> {
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        &mut self.data
    }
}

/// A container that takes a reference to a slice.
pub struct SliceContainer<'a, Item> {
    data: &'a [Item],
}

impl<'a, Item> ContainerTypeHint for SliceContainer<'a, Item> {
    type TypeHint = Heap; // Slice does not have a fixed size, but we can use a marker.
}

impl<'a, Item> SliceContainer<'a, Item> {
    /// New slice container from a reference.
    #[inline(always)]
    pub fn new(slice: &'a [Item]) -> SliceContainer<'a, Item> {
        SliceContainer::<Item> { data: slice }
    }
}

impl<Item> DataContainer for SliceContainer<'_, Item> {
    type Item = Item;

    #[inline(always)]
    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: Copy> ValueDataContainer for SliceContainer<'_, Item> {
    #[inline(always)]
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }
}

impl<Item> RefDataContainer for SliceContainer<'_, Item> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }
}

impl<Item> RawAccessDataContainer for SliceContainer<'_, Item> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.data
    }
}

/// A container that takes a mutable reference to a slice.
pub struct SliceContainerMut<'a, Item> {
    data: &'a mut [Item],
}

impl<'a, Item> ContainerTypeHint for SliceContainerMut<'a, Item> {
    type TypeHint = Heap; // Slice does not have a fixed size, but we can use a marker.
}

impl<'a, Item> SliceContainerMut<'a, Item> {
    /// New mutable slice container from mutable reference.
    pub fn new(slice: &'a mut [Item]) -> SliceContainerMut<'a, Item> {
        SliceContainerMut::<Item> { data: slice }
    }
}

impl<Item> DataContainer for SliceContainerMut<'_, Item> {
    type Item = Item;

    #[inline(always)]
    fn number_of_elements(&self) -> usize {
        self.data.len()
    }
}

impl<Item: Copy> ValueDataContainer for SliceContainerMut<'_, Item> {
    #[inline(always)]
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked(index)
    }
}

impl<Item> RefDataContainer for SliceContainerMut<'_, Item> {
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked(index)
    }
}

impl<Item> RefDataContainerMut for SliceContainerMut<'_, Item> {
    #[inline(always)]
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item {
        debug_assert!(index < self.number_of_elements());
        self.data.get_unchecked_mut(index)
    }
}

impl<Item> ModifiableDataContainer for SliceContainerMut<'_, Item> {
    #[inline(always)]
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item) {
        debug_assert!(index < self.number_of_elements());
        *self.data.get_unchecked_mut(index) = value;
    }
}

impl<Item> RawAccessDataContainer for SliceContainerMut<'_, Item> {
    #[inline(always)]
    fn data(&self) -> &[Self::Item] {
        self.data
    }
}

impl<Item> MutableRawAccessDataContainer for SliceContainerMut<'_, Item> {
    #[inline(always)]
    fn data_mut(&mut self) -> &mut [Self::Item] {
        self.data
    }
}
