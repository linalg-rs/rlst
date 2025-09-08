//! Implementation of operator concepts for dense arrays.
//!
//! This module defines an [ArrayVectorSpace], a concrete implementation of [Space] based on
//! vectors represented through a 1d [Array](crate::dense::array::Array). A [ArrayVectorspace] is also
//! an [InnerProductSpace] and a [NormedSpace](crate::NormedSpace). For the inner product the second vector
//! is taken as complex conjugate.

use std::{marker::PhantomData, ops::Neg};

use crate::{
    EvaluateObject, InnerProductSpace, LinearSpace, RlstScalar, dense::array::DynArray,
    operator::element::Element,
};

/// Array vector space
pub struct ArrayVectorSpace<Item> {
    dimension: usize,
    _marker: PhantomData<Item>,
}

impl<Item> ArrayVectorSpace<Item> {
    /// Create a new array vector space with the given dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            _marker: PhantomData,
        }
    }
}

impl<Item> LinearSpace for ArrayVectorSpace<Item>
where
    Item: Copy
        + Default
        + 'static
        + std::ops::Add<Output = Item>
        + std::ops::Sub<Output = Item>
        + std::ops::Neg<Output = Item>
        + std::ops::AddAssign<Item>
        + std::ops::SubAssign<Item>
        + std::ops::Mul<Item, Output = Item>
        + std::ops::MulAssign<Item>,
{
    type F = Item;

    type Impl = DynArray<Item, 1>;

    fn zero(&self) -> crate::operator::element::Element<Self> {
        Element::new(self, DynArray::<Item, 1>::from_shape([self.dimension]))
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
        Element::new(
            self,
            DynArray::<_, 1>::new_from(&x.imp().r().scalar_mul(*scalar)),
        )
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
        *x.imp_mut() += y.imp().r();
    }

    fn sub_inplace(
        &self,
        x: &mut crate::operator::element::Element<Self>,
        y: &crate::operator::element::Element<Self>,
    ) {
        *x.imp_mut() -= y.imp().r();
    }

    fn scale_inplace(&self, scalar: &Self::F, x: &mut crate::operator::element::Element<Self>) {
        *x.imp_mut() *= *scalar;
    }

    fn copy_from(
        &self,
        x: &crate::operator::element::Element<Self>,
    ) -> crate::operator::element::Element<Self> {
        Element::new(self, x.imp().eval())
    }
}

impl<Item> InnerProductSpace for ArrayVectorSpace<Item>
where
    Item: Copy
        + Default
        + RlstScalar
        + 'static
        + std::ops::Add<Output = Item>
        + std::ops::Sub<Output = Item>
        + std::ops::Neg<Output = Item>
        + std::ops::AddAssign<Item>
        + std::ops::SubAssign<Item>
        + std::ops::Mul<Item, Output = Item>
        + std::ops::MulAssign<Item>,
{
    fn inner_product(&self, x: &Element<Self>, other: &Element<Self>) -> Self::F {
        x.imp().inner(other.imp()).unwrap()
    }
}
