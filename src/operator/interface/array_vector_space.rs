//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;
use std::rc::Rc;

use crate::dense::array::reference::{ArrayRef, ArrayRefMut};
use crate::dense::types::RlstScalar;
use crate::dense::{
    array::{Array, DynamicArray},
    base_array::BaseArray,
    data_container::VectorContainer,
};
use crate::operator::space::{ElementImpl, IndexableSpace, InnerProductSpace, LinearSpace};
use crate::operator::{ConcreteElementContainer, Element};
use crate::rlst_dynamic_array1;
//use crate::{rlst_dynamic_array1, RawAccess, RawAccessMut};

/// Array vector space
pub struct ArrayVectorSpace<Item: RlstScalar> {
    dimension: usize,
    _marker: PhantomData<Item>,
}

/// Element of an array vector space
pub struct ArrayVectorSpaceElement<Item: RlstScalar> {
    elem: DynamicArray<Item, 1>,
    space: Rc<ArrayVectorSpace<Item>>,
}

impl<Item: RlstScalar> ArrayVectorSpaceElement<Item> {
    /// Create a new element
    pub fn new(space: Rc<ArrayVectorSpace<Item>>) -> Self {
        Self {
            elem: rlst_dynamic_array1!(Item, [space.dimension()]),
            space,
        }
    }
}

impl<Item: RlstScalar> Clone for ArrayVectorSpaceElement<Item> {
    fn clone(&self) -> Self {
        let mut new_array = rlst_dynamic_array1!(Item, [self.space.dimension()]);
        new_array.fill_from(self.view().r());
        Self {
            elem: new_array,
            space: self.space.clone(),
        }
    }
}

impl<Item: RlstScalar> ArrayVectorSpace<Item> {
    /// Create a new vector space
    pub fn from_dimension(dimension: usize) -> Rc<Self> {
        Rc::new(Self {
            dimension,
            _marker: PhantomData,
        })
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

    fn zero(space: Rc<Self>) -> Element<ConcreteElementContainer<Self::E>> {
        Element::<ConcreteElementContainer<Self::E>>::new(ArrayVectorSpaceElement::new(space))
    }
}

impl<Item: RlstScalar> InnerProductSpace for ArrayVectorSpace<Item> {
    fn inner_product(&self, x: &Self::E, other: &Self::E) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<Item: RlstScalar> ElementImpl for ArrayVectorSpaceElement<Item> {
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

    fn space(&self) -> Rc<Self::Space> {
        self.space.clone()
    }

    fn view(&self) -> Self::View<'_> {
        self.elem.r()
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        self.elem.r_mut()
    }

    fn axpy_inplace(&mut self, alpha: Self::F, x: &Self) {
        self.elem.sum_into(x.view().scalar_mul(alpha));
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

    fn sub_inplace(&mut self, other: &Self) {
        self.elem.sub_into(other.view());
    }

    /*fn set(&mut self, index: usize, alpha: Self::F) {
        self.view_mut().data_mut()[index] = alpha;
    }

    fn get(&self, index: usize) -> Self::F {
        self.view().data()[index]
    }*/
}
