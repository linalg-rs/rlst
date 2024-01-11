//! A spanning set is a collection of elements of a space.

use crate::{Element, LinearSpace};

/// An indexable collection of elements of a vector space.
///
/// In the simple case of a space of n-vectors a spanning set
/// corresponds to the columns of a matrix.
pub trait SpanningSet {
    type Element: Element;

    /// Get a reference to an element.
    fn get(&self, index: usize) -> &Self::Element;

    /// Get a mutable reference to an element.
    fn get_mut(&mut self, index: usize) -> &mut Self::Element;

    /// Return the number of elements in the spanning set.
    fn nelements(&self) -> usize;

    /// Store a linear combination with coefficients `coeffs` in `result`.
    fn combine(
        &self,
        result: &mut Self::Element,
        coeffs: &[<<Self::Element as Element>::Space as LinearSpace>::F],
    );
}

/// A growable indexable collection of elements of a vector space.
pub trait GrowableSpanningSet: SpanningSet {
    fn append(&mut self, element: Self::Element);
}
