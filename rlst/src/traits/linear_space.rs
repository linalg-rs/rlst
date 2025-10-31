//! Abstract linear spaces and their elements.

use crate::operator::element::Element;

/// Definition of a linear space
///
/// Linear spaces are basic objects that can create
/// elements of the space.
pub trait LinearSpace: Sized {
    /// Field Type.
    type F;

    /// Concrete type associated with elements of the space.
    type Impl;

    /// Create a new zero element from the space.
    fn zero(&self) -> Element<'_, Self>;

    /// Add two elements of the space.
    fn add(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self>;

    /// Subtract two elements of the space.
    fn sub(&self, x: &Element<Self>, y: &Element<Self>) -> Element<'_, Self>;

    /// Multiply an element of the space by a scalar.
    fn scalar_mul(&self, scalar: &Self::F, x: &Element<Self>) -> Element<'_, Self>;

    /// Negate an element of the space.
    fn neg(&self, x: &Element<Self>) -> Element<'_, Self>;

    /// Sum element `y` into  element `x`
    fn sum_inplace(&self, x: &mut Element<Self>, y: &Element<Self>);

    /// Subtract element `y` from element `x`
    fn sub_inplace(&self, x: &mut Element<Self>, y: &Element<Self>);

    /// Multiply with a scalar in place.
    fn scale_inplace(&self, scalar: &Self::F, x: &mut Element<Self>);

    /// Create a new element by copying an existing  one.
    fn copy_from(&self, x: &Element<Self>) -> Element<'_, Self>;
}

/// A dual space
pub trait DualSpace: LinearSpace {
    /// Dual Space
    type DualSpace: LinearSpace<F = Self::F>;

    /// Dual pairing
    fn dual_pairing(&self, x: &Element<Self>, y: &Element<Self::DualSpace>) -> Self::F;
}

/// Indexable space
pub trait IndexableSpace: LinearSpace {
    /// Dimension
    fn dimension(&self) -> usize;
}

/// Inner product space
pub trait InnerProductSpace: LinearSpace {
    /// Inner product
    ///
    /// In spaces over complex numbers it is assumed that the complex conjugate
    /// is taken with respect to the second argument.
    fn inner_product(&self, x: &Element<Self>, y: &Element<Self>) -> Self::F;
}

/// Normed space
pub trait NormedSpace: LinearSpace {
    /// The output type of the norm.
    type Output;

    /// Norm of an element.
    fn norm(&self, x: &Element<Self>) -> Self::Output;
}

/// A frame is a collection of elements of a space.
pub trait Frame {
    ///  The underlying linear space.
    type Space: LinearSpace;
    /// Iterator
    type Iter<'iter>: std::iter::Iterator<Item = &'iter Element<'iter, Self::Space>>
    where
        Self: 'iter;
    /// Mutable iterator
    type IterMut<'iter>: std::iter::Iterator<Item = &'iter mut Element<'iter, Self::Space>>
    where
        Self: 'iter;
    /// Get a reference to an element
    fn get(&self, index: usize) -> Option<&Element<'_, Self::Space>>;
    /// Get a mutable reference to an element
    fn get_mut(&mut self, index: usize) -> Option<&mut Element<'_, Self::Space>>;
    /// Number of elements
    fn len(&self) -> usize;
    /// Is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Get iterator
    fn iter(&self) -> Self::Iter<'_>;
    /// Get mutable iterator
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
    /// Add an element
    fn push(&mut self, elem: Element<Self::Space>);

    /// Remove the last element and return it. If the frame is empty, return None.
    fn pop(&mut self) -> Option<Element<'_, Self::Space>>;
}
