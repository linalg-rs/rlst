//! A spanning set is a collection of elements of a space.

use rlst_common::types::RlstResult;

use crate::{Element, LinearSpace};

/// An indexable collection of elements of a vector space.
///
/// In the simple case of a space of n-vectors a spanning set
/// corresponds to the columns of a matrix.
pub trait Frame {
    type Element: Element;

    /// Get a reference to an element.
    fn get(&self, index: usize) -> Option<&Self::Element>;

    /// Get a mutable reference to an element.
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Element>;

    /// Return the number of elements in the spanning set.
    fn nelements(&self) -> usize;

    /// Store a linear combination with coefficients `coeffs` in `result`.
    fn evaluate(
        &self,
        result: &mut Self::Element,
        coeffs: &[<<Self::Element as Element>::Space as LinearSpace>::F],
    );

    /// Return the associated function space.
    fn space(&self) -> &<Self::Element as Element>::Space;
}

/// A growable indexable collection of elements of a vector space.
pub trait GrowableFrame: Frame {
    /// Add an element to the frame.
    fn extend(&mut self, element: Self::Element) -> RlstResult<()>;
}
