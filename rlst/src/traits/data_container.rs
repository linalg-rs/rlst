//! Traits for data containers.
//!
//! A [DataContainer] is a low-level interface to physical data. Its derived
//! traits [ValueDataContainer], [RefDataContainer], and [RefDataContainerMut]
//! provides value-based access, reference based access, or mutable reference based access.
//! The [ModifiableDataContainer] defines a setter routine that can modify elements by value.
//! By default a data container is not resizeable. To define a resizeable data container implement
//! the [ResizeableDataContainer] trait. For raw access to the underlying data the traits
//! [RawAccessDataContainer] and [RawAccessDataContainerMut] are provided.
//! The [ContainerType] trait attaches an associated type that can be used to distinguish between containers
//! on a type level.

/// Base trait for a data container. It defines the type of the underlying data
/// and provides a method to return the number of elements.
pub trait DataContainer {
    /// Item type
    type Item: Copy + Default;

    /// Return the number of elements in the container.
    fn number_of_elements(&self) -> usize;
}

/// A data container that provides value based access.
pub trait ValueDataContainer: DataContainer {
    /// Access the container unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item;
}

/// A data container that provides reference based access.
pub trait RefDataContainer: DataContainer {
    /// Access the container by reference
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;
}

/// A data container that provides access by mutable reference.
pub trait RefDataContainerMut: DataContainer {
    /// Access the container mutably unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// A data container that can modify data by value.
pub trait ModifiableDataContainer: DataContainer {
    /// Set the value at a given index.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item);
}

/// A resizeable data container that can change its size at runtime.
pub trait ResizeableDataContainer: DataContainer {
    /// Resize the data container by providing the new length in `new_len`.
    fn resize(&mut self, new_len: usize);
}

/// A data container that provides raw access to the data slice.
pub trait RawAccessDataContainer: DataContainer {
    /// Return the data slice for the data container.
    fn data(&self) -> &[Self::Item];
}

/// A data container that provides mutable raw access to the data slice.
pub trait MutableRawAccessDataContainer: RawAccessDataContainer {
    /// Return the mutable data slice for the data container.
    fn data_mut(&mut self) -> &mut [Self::Item];
}

/// Stores the type of a container.
pub trait ContainerType {
    /// The type for the container.
    type Type: ContainerTypeRepr;

    /// Returns the type hint for the container as string ref.
    fn type_as_str(&self) -> &'static str {
        Self::Type::STR
    }
}

/// Associate a string representation with a container type.
pub trait ContainerTypeRepr {
    /// The string representation of the container type.
    const STR: &str;
}

/// Selects a container type based on two other container types.
pub trait ContainerTypeSelector<U: ContainerTypeRepr, V: ContainerTypeRepr> {
    /// Select the container type.
    type Type: ContainerTypeRepr;
}
