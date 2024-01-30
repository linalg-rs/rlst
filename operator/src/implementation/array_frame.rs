use crate::space::*;
use rlst_common::types::Scalar;

use self::frame::{Frame, GrowableFrame};

use super::array_vector_space::{ArrayVectorSpace, ArrayVectorSpaceElement};

pub struct ArrayFrame<'space, Item: Scalar> {
    data: Vec<ArrayVectorSpaceElement<'space, Item>>,
    space: &'space ArrayVectorSpace<Item>,
}

impl<'space, Item: Scalar> ArrayFrame<'space, Item> {}

impl<'space, Item: Scalar> Frame for ArrayFrame<'space, Item> {
    type Element = ArrayVectorSpaceElement<'space, Item>;

    fn get(&self, index: usize) -> Option<&Self::Element> {
        self.data.get(index)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Element> {
        self.data.get_mut(index)
    }

    fn nelements(&self) -> usize {
        self.data.len()
    }

    fn evaluate(
        &self,
        result: &mut Self::Element,
        coeffs: &[<<Self::Element as Element>::Space as LinearSpace>::F],
    ) {
        assert_eq!(self.nelements(), coeffs.len());
        assert!(self.space().is_same(result.space()));
    }

    fn space(&self) -> &<Self::Element as Element>::Space {
        self.space
    }
}

impl<'space, Item: Scalar> GrowableFrame for ArrayFrame<'space, Item> {
    fn extend(&mut self, element: Self::Element) -> rlst_common::types::RlstResult<()> {
        self.data.push(element);
        Ok(())
    }
}
