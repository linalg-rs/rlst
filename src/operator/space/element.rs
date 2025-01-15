//! Elements of linear spaces
use std::{
    borrow::Borrow,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::dense::types::RlstScalar;
use num::One;

use super::{InnerProductSpace, LinearSpace, NormedSpace};

/// An Element of a linear spaces.
pub trait ElementImpl {
    /// Space type
    type Space: LinearSpace<F = Self::F, E = Self>;
    /// Scalar Type
    type F: RlstScalar;
    /// View
    type View<'b>
    where
        Self: 'b;
    /// Mutable view
    type ViewMut<'b>
    where
        Self: 'b;

    /// Return the associated function space.
    fn space(&self) -> Rc<Self::Space>;

    /// Get a view onto the element.
    fn view(&self) -> Self::View<'_>;

    /// Get a mutable view onto the element.
    fn view_mut(&mut self) -> Self::ViewMut<'_>;

    /// self += alpha * other.
    fn axpy_inplace(&mut self, alpha: Self::F, other: &Self);

    /// self += other.
    fn sum_inplace(&mut self, other: &Self) {
        self.axpy_inplace(<Self::F as One>::one(), other);
    }

    /// self = other.
    fn fill_inplace(&mut self, other: &Self);

    /// self *= alpha.
    fn scale_inplace(&mut self, alpha: Self::F);

    /// self = -self.
    fn neg_inplace(&mut self) {
        self.scale_inplace(-<Self::F as One>::one());
    }

    // /// self += alpha * other.
    // fn axpy(mut self, alpha: Self::F, other: &Self) -> Self
    // where
    //     Self: Sized,
    // {
    //     self.axpy_inplace(alpha, other);
    //     self
    // }

    // /// self += other
    // fn sum(mut self, other: &Self) -> Self
    // where
    //     Self: Sized,
    // {
    //     self.sum_inplace(other);
    //     self
    // }

    // /// self = other
    // fn fill(mut self, other: &Self) -> Self
    // where
    //     Self: Sized,
    // {
    //     self.fill_inplace(other);
    //     self
    // }

    // /// self = alpha * self
    // fn scale(mut self, alpha: Self::F) -> Self
    // where
    //     Self: Sized,
    // {
    //     self.scale_inplace(alpha);
    //     self
    // }

    // /// self = -self
    // fn neg(mut self) -> Self
    // where
    //     Self: Sized,
    // {
    //     self.neg_inplace();
    //     self
    // }
}

/// The view type associated with elements of linear spaces.
pub type ElementView<'view, Space> = <<Space as LinearSpace>::E as ElementImpl>::View<'view>;

/// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'view, Space> = <<Space as LinearSpace>::E as ElementImpl>::ViewMut<'view>;

pub trait ElementContainer {
    type E: ElementImpl;

    fn imp(&self) -> &Self::E;

    fn space(&self) -> Rc<<Self::E as ElementImpl>::Space> {
        self.imp().space()
    }

    fn view(&self) -> ElementView<'_, <Self::E as ElementImpl>::Space> {
        self.imp().view()
    }

    /// Comppute the inner product with another vector
    ///
    /// Only implemented for elements of inner product spaces.
    fn inner_product(
        &self,
        other: impl ElementContainer<E = Self::E>,
    ) -> <Self::E as ElementImpl>::F
    where
        <Self::E as ElementImpl>::Space: super::InnerProductSpace,
    {
        self.space().inner_product(self.imp(), other.imp())
    }

    /// Compute the norm of a vector
    ///
    /// Only implemented for elements of normed spaces.
    fn norm(&self) -> <<Self::E as ElementImpl>::F as RlstScalar>::Real
    where
        <Self::E as ElementImpl>::Space: super::NormedSpace,
    {
        self.space().norm(self.imp())
    }
}

pub trait ElementContainerMut: ElementContainer {
    fn imp_mut(&mut self) -> &mut Self::E;

    fn view_mut(&mut self) -> ElementViewMut<'_, <Self::E as ElementImpl>::Space> {
        self.imp_mut().view_mut()
    }

    fn axpy_inplace(
        &mut self,
        alpha: <Self::E as ElementImpl>::F,
        other: impl ElementContainer<E = Self::E>,
    ) {
        self.imp_mut().axpy_inplace(alpha, other.imp());
    }

    fn sum_inplace(&mut self, other: Element<impl ElementContainer<E = Self::E>>) {
        self.imp_mut().sum_inplace(other.imp());
    }

    fn fill_inplace(&mut self, other: Element<impl ElementContainer<E = Self::E>>) {
        self.imp_mut().fill_inplace(other.imp());
    }

    fn scale_inplace(&mut self, alpha: <Self::E as ElementImpl>::F) {
        self.imp_mut().scale_inplace(alpha);
    }
}

pub struct ConcreteElementContainer<ElemImpl: ElementImpl>(ElemImpl);

pub struct ConcreteElementContainerRef<'a, ElemImpl: ElementImpl> {
    elem: &'a ElemImpl,
}

pub struct ConcreteElementContainerRefMut<'a, ElemImpl: ElementImpl> {
    elem: &'a mut ElemImpl,
}

impl<ElemImpl: ElementImpl> ElementContainer for ConcreteElementContainer<ElemImpl> {
    type E = ElemImpl;

    fn imp(&self) -> &Self::E {
        &self.0
    }
}

impl<ElemImpl: ElementImpl> ElementContainer for ConcreteElementContainerRef<'_, ElemImpl> {
    type E = ElemImpl;

    fn imp(&self) -> &Self::E {
        self.elem
    }
}

impl<ElemImpl: ElementImpl> ElementContainer for ConcreteElementContainerRefMut<'_, ElemImpl> {
    type E = ElemImpl;

    fn imp(&self) -> &Self::E {
        self.elem
    }
}

impl<ElemImpl: ElementImpl> ElementContainerMut for ConcreteElementContainerRefMut<'_, ElemImpl> {
    fn imp_mut(&mut self) -> &mut Self::E {
        self.elem
    }
}

pub struct Element<Container: ElementContainer>(Container);

impl<Container: ElementContainer> Element<Container> {
    pub fn imp(&self) -> &Container::E {
        self.0.imp()
    }
}

impl<Container: ElementContainerMut> Element<Container> {
    pub fn imp_mut(&mut self) -> &mut Container::E {
        self.0.imp_mut()
    }
}

impl<ElemImpl: ElementImpl> Element<ConcreteElementContainer<ElemImpl>> {
    pub fn new(elem: ElemImpl) -> Self {
        Self(ConcreteElementContainer(elem))
    }
}

impl<ElemImpl: ElementImpl> Deref for Element<ConcreteElementContainer<ElemImpl>> {
    type Target = ConcreteElementContainer<ElemImpl>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, ElemImpl: ElementImpl> Element<ConcreteElementContainerRef<'a, ElemImpl>> {
    pub fn new(elem: &'a ElemImpl) -> Self {
        Self(ConcreteElementContainerRef { elem })
    }
}

impl<'a, ElemImpl: ElementImpl> Deref for Element<ConcreteElementContainerRef<'a, ElemImpl>> {
    type Target = ConcreteElementContainerRef<'a, ElemImpl>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, ElemImpl: ElementImpl> Element<ConcreteElementContainerRefMut<'a, ElemImpl>> {
    pub fn new(elem: &'a mut ElemImpl) -> Self {
        Self(ConcreteElementContainerRefMut { elem })
    }
}

impl<'a, ElemImpl: ElementImpl> Deref for Element<ConcreteElementContainerRefMut<'a, ElemImpl>> {
    type Target = ConcreteElementContainerRefMut<'a, ElemImpl>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, ElemImpl: ElementImpl> DerefMut for Element<ConcreteElementContainerRefMut<'a, ElemImpl>> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
