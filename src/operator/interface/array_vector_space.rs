//! Implementation of operator concepts for dense arrays.

use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, MulAssign, Neg},
};

use crate::{
    dense::{
        array::{operators::addition::ArrayAddition, reference::ArrayRef, DynArray, RefType},
        base_array::BaseArray,
        data_container::VectorContainer,
    },
    operator::element::Element,
    Array, AsRefType, BaseItem, EvaluateArray, Inner, InnerProductSpace, LinearSpace, ScalarMul,
    ScaleInPlace,
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
        Element::new(self, (x.imp().r().scalar_mul(*scalar)).eval())
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
        + 'static
        + std::ops::Add<Output = Item>
        + std::ops::Sub<Output = Item>
        + std::ops::Neg<Output = Item>
        + std::ops::AddAssign<Item>
        + std::ops::Mul<Item, Output = Item>
        + std::ops::MulAssign<Item>,
    DynArray<Item, 1>: Inner<DynArray<Item, 1>, Output = Item>,
{
    fn inner_product(&self, x: &Element<Self>, other: &Element<Self>) -> Self::F {
        x.imp().inner(other.imp())
    }
}
