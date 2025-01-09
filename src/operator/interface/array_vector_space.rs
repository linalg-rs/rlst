//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;

use crate::dense::array::reference::{ArrayRef, ArrayRefMut};
use crate::dense::types::RlstScalar;
use crate::dense::{
    array::{Array, DynamicArray},
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
pub struct ArrayVectorSpaceElement<'a, Item: RlstScalar> {
    elem: DynamicArray<Item, 1>,
    space: &'a ArrayVectorSpace<Item>,
}

impl<'a, Item: RlstScalar> ArrayVectorSpaceElement<'a, Item> {
    /// Create a new element
    pub fn new(space: &'a ArrayVectorSpace<Item>) -> Self {
        Self {
            elem: rlst_dynamic_array1!(Item, [space.dimension()]),
            space,
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
    type E<'elem>
        = ArrayVectorSpaceElement<'elem, Item>
    where
        Self: 'elem;

    type F = Item;

    fn zero(&self) -> Self::E<'_> {
        ArrayVectorSpaceElement::new(self)
    }
}

impl<Item: RlstScalar> InnerProductSpace for ArrayVectorSpace<Item> {
    fn inner<'a>(&self, x: &Self::E<'a>, other: &Self::E<'a>) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<'a, Item: RlstScalar> Element<'a> for ArrayVectorSpaceElement<'a, Item> {
    type F = Item;
    type Space = ArrayVectorSpace<Item>;

    type View<'b>
        = Array<Item, ArrayRef<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1>
    where
        Self: 'b;

    type ViewMut<'b>
        = Array<Item, ArrayRefMut<'b, Item, BaseArray<Item, VectorContainer<Item>, 1>, 1>, 1>
    where
        Self: 'b;

    fn space(&self) -> &Self::Space {
        self.space
    }

    fn view(&self) -> Self::View<'_> {
        self.elem.r()
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.elem.r_mut()
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

impl<Item: RlstScalar> Clone for ArrayVectorSpaceElement<'_, Item> {
    fn clone(&self) -> Self {
        let mut new_array = rlst_dynamic_array1!(Item, [self.space.dimension()]);
        new_array.fill_from(self.elem.r());
        Self {
            elem: new_array,
            space: self.space,
        }
    }
}
