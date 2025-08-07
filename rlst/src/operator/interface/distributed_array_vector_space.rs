//! Implementation of operator concepts for dense arrays.

use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    rc::Rc,
};

use mpi::traits::{Communicator, Equivalence};

use crate::{
    dense::{base_array::BaseArray, data_container::VectorContainer},
    dist_vec,
    distributed_tools::IndexLayout,
    operator::element::Element,
    sparse::distributed_array::DistributedArray,
    EvaluateObject, Inner, InnerProductSpace, LinearSpace,
};

/// Array vector space
pub struct DistributedArrayVectorSpace<'a, C: Communicator, Item: Equivalence> {
    index_layout: Rc<IndexLayout<'a, C>>,
    _marker: PhantomData<Item>,
}

impl<'a, C: Communicator, Item: Equivalence> DistributedArrayVectorSpace<'a, C, Item> {
    /// Create a new vector space
    pub fn new(index_layout: Rc<IndexLayout<'a, C>>) -> Self {
        Self {
            index_layout,
            _marker: PhantomData,
        }
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

impl<'a, C, Item> LinearSpace for DistributedArrayVectorSpace<'a, C, Item>
where
    C: Communicator,
    Item: Equivalence
        + Copy
        + Default
        + Add<Item, Output = Item>
        + Sub<Item, Output = Item>
        + Mul<Item, Output = Item>
        + Neg<Output = Item>
        + AddAssign<Item>
        + SubAssign<Item>
        + MulAssign<Item>,
{
    type F = Item;

    type Impl = DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>;

    fn zero(&self) -> crate::operator::element::Element<Self> {
        Element::new(self, dist_vec!(Item, self.index_layout.clone()))
    }

    fn add(
        &self,
        x: &crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, (x.imp().r() + y.imp().r()).eval())
    }

    fn sub(
        &self,
        x: &crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, (x.imp().r() - y.imp().r()).eval())
    }

    fn scalar_mul(
        &self,
        scalar: &Self::F,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, x.imp().r().scalar_mul(*scalar).eval())
    }

    fn neg(
        &self,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, x.imp().r().neg().eval())
    }

    fn sum_inplace(
        &self,
        x: &mut crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) {
        x.imp_mut().r_mut().add_assign(y.imp().r());
    }

    fn sub_inplace(
        &self,
        x: &mut crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) {
        x.imp_mut().r_mut().sub_assign(y.imp().r());
    }

    fn scale_inplace(&self, scalar: &Self::F, x: &mut crate::operator::element::Element<Self>) {
        x.imp_mut().r_mut().mul_assign(*scalar);
    }

    fn copy_from(
        &self,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, x.imp().eval())
    }
}

impl<'a, C, Item> InnerProductSpace for DistributedArrayVectorSpace<'a, C, Item>
where
    C: Communicator,
    Item: Equivalence
        + Copy
        + Default
        + Add<Item, Output = Item>
        + Sub<Item, Output = Item>
        + Mul<Item, Output = Item>
        + Neg<Output = Item>
        + AddAssign<Item>
        + SubAssign<Item>
        + MulAssign<Item>,
    DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>:
        Inner<DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>, Output = Item>,
{
    fn inner_product(&self, x: &Element<Self>, other: &Element<Self>) -> Self::F {
        x.imp().inner(other.imp())
    }
}
