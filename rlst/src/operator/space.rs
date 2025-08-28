//! Linear spaces and their elements.

use crate::{Inner, InnerProductSpace, LinearSpace, NormedSpace, RlstScalar};

use super::element::Element;

impl<Item: RlstScalar, S: InnerProductSpace<F = Item>> NormedSpace for S {
    type Output = <Item as RlstScalar>::Real;

    fn norm(&self, x: &super::element::Element<Self>) -> Self::Output {
        x.inner(x).abs().sqrt()
    }
}

/// Return the zero element of a given space.
pub fn zero_element<Space: LinearSpace>(space: &Space) -> Element<Space> {
    space.zero()
}
