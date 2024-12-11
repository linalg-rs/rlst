//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;

use mpi::traits::{Communicator, Equivalence};

use crate::dense::types::RlstScalar;
use crate::operator::space::{Element, IndexableSpace, InnerProductSpace, LinearSpace};
use crate::{DefaultDistributedIndexLayout, DistributedVector};
use bempp_distributed_tools::IndexLayout;

/// Array vector space
pub struct DistributedArrayVectorSpace<'a, Item: RlstScalar + Equivalence, C: Communicator> {
    index_layout: &'a DefaultDistributedIndexLayout<'a, C>,
    _marker: PhantomData<Item>,
}

/// Element of an array vector space
pub struct DistributedArrayVectorSpaceElement<'a, Item: RlstScalar + Equivalence, C: Communicator> {
    elem: DistributedVector<'a, Item, C>,
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence>
    DistributedArrayVectorSpaceElement<'a, Item, C>
{
    /// Create a new element
    pub fn new(space: &DistributedArrayVectorSpace<'a, Item, C>) -> Self {
        Self {
            elem: DistributedVector::new(space.index_layout),
        }
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> DistributedArrayVectorSpace<'a, Item, C> {
    /// Create a new vector space
    pub fn new(index_layout: &'a DefaultDistributedIndexLayout<'a, C>) -> Self {
        Self {
            index_layout,
            _marker: PhantomData,
        }
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence> IndexableSpace
    for DistributedArrayVectorSpace<'_, Item, C>
{
    fn dimension(&self) -> usize {
        self.index_layout.number_of_global_indices()
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> LinearSpace
    for DistributedArrayVectorSpace<'a, Item, C>
{
    type E = DistributedArrayVectorSpaceElement<'a, Item, C>;

    type F = Item;

    fn zero(&self) -> Self::E {
        DistributedArrayVectorSpaceElement::new(self)
    }
}

impl<C: Communicator, Item: RlstScalar + Equivalence> InnerProductSpace
    for DistributedArrayVectorSpace<'_, Item, C>
{
    fn inner(&self, x: &Self::E, other: &Self::E) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<'a, C: Communicator, Item: RlstScalar + Equivalence> Element
    for DistributedArrayVectorSpaceElement<'a, Item, C>
{
    type F = Item;
    type Space = DistributedArrayVectorSpace<'a, Item, C>;

    type View<'b>
        = &'b DistributedVector<'a, Item, C>
    where
        Self: 'b;

    type ViewMut<'b>
        = &'b mut DistributedVector<'a, Item, C>
    where
        Self: 'b;

    fn view(&self) -> Self::View<'_> {
        &self.elem
    }

    fn view_mut(&mut self) -> Self::ViewMut<'_> {
        &mut self.elem
    }

    fn axpy_inplace(&mut self, alpha: Self::F, other: &Self) {
        //self.elem.sum_into(other.r().scalar_mul(alpha));
        self.elem
            .local_mut()
            .sum_into(other.view().local().r().scalar_mul(alpha));
    }

    fn sum_inplace(&mut self, other: &Self) {
        self.elem.local_mut().sum_into(other.view().local().r());
    }

    fn fill_inplace(&mut self, other: &Self) {
        self.elem.local_mut().fill_from(other.view().local().r());
    }

    fn scale_inplace(&mut self, alpha: Self::F) {
        self.elem.local_mut().scale_inplace(alpha);
    }
}
