//! Traits for data containers.

/// Defines the basic behaviour of a data container.
pub trait DataContainer {
    /// Item type
    type Item;

    /// Return the number of elements in the container.
    fn number_of_elements(&self) -> usize;
}

/// A Container that can return values.
pub trait ValueDataContainer: DataContainer {
    /// Access the container unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_value(&self, index: usize) -> Self::Item;
}

/// A container that can return references.
pub trait RefDataContainer: DataContainer {
    /// Access the container by reference
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::Item;
}

/// Mutable data container.
pub trait RefDataContainerMut: DataContainer {
    /// Access the container mutably unchecked.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::Item;
}

/// A container that can modify data.
pub trait ModifiableDataContainer: DataContainer {
    /// Set the value at a given index.
    ///
    /// # Safety
    /// `index` must not be out of bounds.
    unsafe fn set_unchecked_value(&mut self, index: usize, value: Self::Item);
}

/// Definition of a resizeable data container.
///
/// A resizeable data container can change its size during runtime.
pub trait ResizeableDataContainer: DataContainer {
    /// Resize the data container.
    fn resize(&mut self, new_len: usize);
}

/// Definition of a data container that allows raw access.
pub trait RawAccessDataContainer: DataContainer {
    /// Return a raw pointer to the data.
    fn data(&self) -> &[Self::Item];
}

/// Definition of a data container that allows raw access.
pub trait MutableRawAccessDataContainer: RawAccessDataContainer {
    /// Return a raw pointer to the data.
    fn data_mut(&mut self) -> &mut [Self::Item];
}

/// Stores the type of a container.
pub trait ContainerTypeHint {
    /// The type hint for the container.
    type TypeHint: ContainerType;

    /// Returns the type hint for the container as string ref.
    fn type_hint_as_str(&self) -> &'static str {
        Self::TypeHint::STR
    }
}

/// Marker trait for container types.
pub trait ContainerType {
    /// The string representation of the container type hint.
    const STR: &str;
}

/// Selects a container type based on two other container types.
pub trait ContainerTypeSelector<U: ContainerType, V: ContainerType> {
    /// Select the container type.
    type Type: ContainerType;
}
