//! Implementation of operator concepts for dense arrays.

use std::marker::PhantomData;

use mpi::traits::Equivalence;

use crate::dense::types::RlstScalar;
use crate::operator::space::{Element, IndexableSpace, InnerProductSpace, LinearSpace};
use crate::DistributedVector;
use bempp_distributed_tools::IndexLayout;

/// Array vector space
pub struct DistributedArrayVectorSpace<Layout: IndexLayout, Item: RlstScalar + Equivalence> {
    index_layout: Layout,
    _marker: PhantomData<Item>,
}

/// Element of an array vector space
pub struct DistributedArrayVectorSpaceElement<
    'a,
    Layout: IndexLayout,
    Item: RlstScalar + Equivalence,
> {
    elem: DistributedVector<'a, Layout, Item>,
    space: &'a DistributedArrayVectorSpace<Layout, Item>,
}

impl<'a, Layout: IndexLayout, Item: RlstScalar + Equivalence>
    DistributedArrayVectorSpaceElement<'a, Layout, Item>
{
    /// Create a new element
    pub fn new(space: &'a DistributedArrayVectorSpace<Layout, Item>) -> Self {
        Self {
            elem: DistributedVector::new(&space.index_layout),
            space,
        }
    }
}

impl<Layout: IndexLayout, Item: RlstScalar + Equivalence>
    DistributedArrayVectorSpace<Layout, Item>
{
    /// Create a new vector space
    pub fn new(index_layout: Layout) -> Self {
        Self {
            index_layout,
            _marker: PhantomData,
        }
    }

    /// Return the communicator
    pub fn comm(&self) -> &Layout::Comm {
        self.index_layout.comm()
    }

    /// Return the index layout
    pub fn index_layout(&self) -> &Layout {
        &self.index_layout
    }
}

impl<Layout: IndexLayout, Item: RlstScalar + Equivalence> IndexableSpace
    for DistributedArrayVectorSpace<Layout, Item>
{
    fn dimension(&self) -> usize {
        self.index_layout.number_of_global_indices()
    }
}

impl<Layout: IndexLayout, Item: RlstScalar + Equivalence> LinearSpace
    for DistributedArrayVectorSpace<Layout, Item>
{
    type E<'a>
        = DistributedArrayVectorSpaceElement<'a, Layout, Item>
    where
        Self: 'a;

    type F = Item;

    fn zero(&self) -> Self::E<'_> {
        DistributedArrayVectorSpaceElement::new(self)
    }
}

impl<Layout: IndexLayout, Item: RlstScalar + Equivalence> InnerProductSpace
    for DistributedArrayVectorSpace<Layout, Item>
{
    fn inner<'a>(&self, x: &'a Self::E<'a>, other: &'a Self::E<'a>) -> Self::F {
        x.view().inner(other.view())
    }
}

impl<'a, Layout: IndexLayout, Item: RlstScalar + Equivalence> Element<'a>
    for DistributedArrayVectorSpaceElement<'a, Layout, Item>
{
    type F = Item;
    type Space = DistributedArrayVectorSpace<Layout, Item>;

    type View<'b>
        = &'b DistributedVector<'a, Layout, Item>
    where
        Self: 'b;

    type ViewMut<'b>
        = &'b mut DistributedVector<'a, Layout, Item>
    where
        Self: 'b;

    fn space(&self) -> &Self::Space {
        &self.space
    }

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

impl<'a, Layout: IndexLayout, Item: RlstScalar + Equivalence> Clone
    for DistributedArrayVectorSpaceElement<'a, Layout, Item>
{
    fn clone(&self) -> Self {
        let mut elem = DistributedArrayVectorSpaceElement::new(self.space);
        elem.view().local().fill_from(self.view().local().r());
        elem
    }
}
