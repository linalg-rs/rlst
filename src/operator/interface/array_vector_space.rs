//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;

use crate::dense::types::RlstScalar;
use crate::dense::{
    array::{
        views::{ArrayView, ArrayViewMut},
        Array, DynamicArray,
    },
    base_array::BaseArray,
    data_container::VectorContainer,
};
use crate::operator::space::{Element, IndexableSpace, InnerProductSpace, LinearSpace};
use crate::rlst_dynamic_array1;

/// Array vector space
pub struct ArrayVectorSpace<Item: RlstScalar> {
    dimension: usize,
    _marker: PhantomData<Item>,
}

/// Element of an array vector space
pub struct ArrayVectorSpaceElement<Item: RlstScalar> {
    elem: DynamicArray<Item, 1>,
}

impl<Item: RlstScalar> ArrayVectorSpaceElement<Item> {
    /// Create a new element
    pub fn new(space: &ArrayVectorSpace<Item>) -> Self {
        Self {
            elem: rlst_dynamic_array1!(Item, [space.dimension()]),
        }
    }
}

impl<Item: RlstScalar> ArrayVectorSpace<Item> {
    /// Create a new vector space
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            _marker: PhantomData,
        }
    }
}

impl<Item: RlstScalar> IndexableSpace for ArrayVectorSpace<Item> {
    fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<Item: RlstScalar> LinearSpace for ArrayVectorSpace<Item> {
    type E = ArrayVectorSpaceElement<Item>;

    type F = Item;

    fn zero(&self) -> Self::E {
        ArrayVectorSpaceElement::new(self)
    }
}

impl<Item: RlstScalar> InnerProductSpace for ArrayVectorSpace<Item> {
    fn inner(&self, x: &Self::E, other: &Self::E) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<Item: RlstScalar> Element for ArrayVectorSpaceElement<Item> {
    type F = Item;
    type Space = ArrayVectorSpace<Item>;

    type View<'b>
        = Array<Item, ArrayView<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1>
    where
        Self: 'b;

    type ViewMut<'b>
        = Array<Item, ArrayViewMut<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1>
    where
        Self: 'b;

    fn view(&self) -> Self::View<'_> {
        self.elem.view()
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.elem.view_mut()
    }

    fn axpy_inplace(&mut self, alpha: Self::F, other: &Self) {
        self.elem.sum_into(other.view().scalar_mul(alpha));
    }

    fn sum_inplace(&mut self, other: &Self) {
        self.elem.sum_into(other.view());
    }

    fn fill_inplace(&mut self, other: &Self) {
        self.elem.fill_from(other.view());
    }

    fn scale_inplace(&mut self, alpha: Self::F) {
        self.view_mut().scale_inplace(alpha);
    }
}
