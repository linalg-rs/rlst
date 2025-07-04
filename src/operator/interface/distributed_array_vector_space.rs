//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;
use std::rc::Rc;

use mpi::traits::{Communicator, Equivalence};

use crate::dense::types::RlstScalar;
use crate::operator::space::{ElementImpl, IndexableSpace, InnerProductSpace, LinearSpace};
use crate::operator::{ConcreteElementContainer, Element};
use crate::DistributedVector;
use bempp_distributed_tools::IndexLayout;

/// Array vector space
pub struct DistributedArrayVectorSpace<'a, C: Communicator, Item: RlstScalar + Equivalence> {
    index_layout: Rc<IndexLayout<'a, C>>,
    _marker: PhantomData<Item>,
}

/// Element of an array vector space
pub struct DistributedArrayVectorSpaceElement<'a, C: Communicator, Item: RlstScalar + Equivalence> {
    elem: DistributedVector<'a, C, Item>,
    space: Rc<DistributedArrayVectorSpace<'a, C, Item>>,
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence>
    DistributedArrayVectorSpaceElement<'a, C, Item>
{
    /// Create a new element
    pub fn new(space: Rc<DistributedArrayVectorSpace<'a, C, Item>>) -> Self {
        Self {
            elem: DistributedVector::new(space.index_layout.clone()),
            space: space.clone(),
        }
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> DistributedArrayVectorSpace<'a, C, Item> {
    /// Create a new vector space
    pub fn from_index_layout(index_layout: Rc<IndexLayout<'a, C>>) -> Rc<Self> {
        Rc::new(Self {
            index_layout,
            _marker: PhantomData,
        })
    }

    /// Return the communicator
    pub fn comm(&self) -> &C {
        self.index_layout.comm()
    }

    /// Return the index layout
    pub fn index_layout(&self) -> Rc<IndexLayout<'a, C>> {
        self.index_layout.clone()
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence> IndexableSpace
    for DistributedArrayVectorSpace<'_, C, Item>
{
    fn dimension(&self) -> usize {
        self.index_layout.number_of_global_indices()
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> LinearSpace
    for DistributedArrayVectorSpace<'a, C, Item>
{
    type E = DistributedArrayVectorSpaceElement<'a, C, Item>;

    type F = Item;

    fn zero(space: Rc<Self>) -> Element<ConcreteElementContainer<Self::E>> {
        Element::<ConcreteElementContainer<_>>::new(DistributedArrayVectorSpaceElement::new(
            space.clone(),
        ))
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence> InnerProductSpace
    for DistributedArrayVectorSpace<'_, C, Item>
{
    fn inner_product(&self, x: &Self::E, other: &Self::E) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> ElementImpl
    for DistributedArrayVectorSpaceElement<'a, C, Item>
{
    type F = Item;
    type Space = DistributedArrayVectorSpace<'a, C, Item>;

    type View<'b>
        = &'b DistributedVector<'a, C, Item>
    where
        Self: 'b;

    type ViewMut<'b>
        = &'b mut DistributedVector<'a, C, Item>
    where
        Self: 'b;

    fn space(&self) -> Rc<Self::Space> {
        self.space.clone()
    }

    fn view(&self) -> Self::View<'_> {
        &self.elem
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        &mut self.elem
    }

    fn axpy_inplace(&mut self, alpha: Self::F, x: &Self) {
        //self.elem.sum_into(other.r().scalar_mul(alpha));
        self.elem
            .local_mut()
            .sum_into(x.view().local().r().scalar_mul(alpha));
    }

    fn sum_inplace(&mut self, other: &Self) {
        self.elem.local_mut().sum_into(other.view().local().r());
    }

    fn fill_inplace(&mut self, other: &Self) {
        self.elem.local_mut().fill_from(other.view().local().r());
    }

    fn fill_inplace_raw(){
        self.elem.local_mut().fill_from_raw_data(other);
    }

    fn scale_inplace(&mut self, alpha: Self::F) {
        self.elem.local_mut().scale_inplace(alpha);
    }

    fn sub_inplace(&mut self, other: &Self) {
        self.elem.local_mut().sub_into(other.view().local().r());
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence> Clone
    for DistributedArrayVectorSpaceElement<'_, C, Item>
{
    fn clone(&self) -> Self {
        let mut elem = DistributedArrayVectorSpaceElement::new(self.space.clone());
        elem.view_mut()
            .local_mut()
            .fill_from(self.view().local().r());
        elem
    }
}
