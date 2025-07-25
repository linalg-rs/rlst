//! Linear spaces and their elements.

use crate::{Inner, InnerProductSpace, NormedSpace, RlstScalar};

impl<Item: RlstScalar, S: InnerProductSpace<F = Item>> NormedSpace for S {
    type Output = <Item as RlstScalar>::Real;

    fn norm(&self, x: &super::element::Element<Self>) -> Self::Output {
        x.inner(x).abs().sqrt()
    }
}
