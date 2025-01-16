//! Elements of linear spaces
use std::{
    borrow::Borrow,
    ops::{Deref, DerefMut},
    rc::Rc,
};

use crate::dense::types::{c32, c64};

use crate::dense::types::RlstScalar;
use num::One;

use super::{zero_element, InnerProductSpace, LinearSpace, NormedSpace};

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

    /// self -= other.
    fn sub_inplace(&mut self, other: &Self);

    /// self = other.
    fn fill_inplace(&mut self, other: &Self);

    /// self *= alpha.
    fn scale_inplace(&mut self, alpha: Self::F);
}

/// The view type associated with elements of linear spaces.
pub type ElementView<'view, Space> = <<Space as LinearSpace>::E as ElementImpl>::View<'view>;

/// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'view, Space> = <<Space as LinearSpace>::E as ElementImpl>::ViewMut<'view>;

pub trait ElementContainer {
    type E: ElementImpl;

    fn imp(&self) -> &Self::E;
}

pub trait ElementContainerMut: ElementContainer {
    fn imp_mut(&mut self) -> &mut Self::E;
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

impl<ElemImpl: ElementImpl> ElementContainerMut for ConcreteElementContainer<ElemImpl> {
    fn imp_mut(&mut self) -> &mut Self::E {
        &mut self.0
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

    pub fn r(&self) -> Element<ConcreteElementContainerRef<'_, Container::E>> {
        Element(ConcreteElementContainerRef { elem: self.imp() })
    }

    pub fn duplicate(&self) -> Element<ConcreteElementContainer<Container::E>> {
        let mut x = zero_element(self.space());

        x.fill_inplace(self.r());

        x
    }

    pub fn space(&self) -> Rc<<Container::E as ElementImpl>::Space> {
        self.0.imp().space()
    }

    pub fn view(&self) -> ElementView<'_, <Container::E as ElementImpl>::Space> {
        self.0.imp().view()
    }

    pub fn inner_product(
        &self,
        other: Element<impl ElementContainer<E = Container::E>>,
    ) -> <Container::E as ElementImpl>::F
    where
        <Container::E as ElementImpl>::Space: InnerProductSpace,
    {
        self.0
            .imp()
            .space()
            .inner_product(self.0.imp(), other.0.imp())
    }

    pub fn norm(&self) -> <<Container::E as ElementImpl>::F as RlstScalar>::Real
    where
        <Container::E as ElementImpl>::Space: NormedSpace,
    {
        self.0.imp().space().norm(self.0.imp())
    }
}

impl<Container: ElementContainerMut> Element<Container> {
    pub fn imp_mut(&mut self) -> &mut Container::E {
        self.0.imp_mut()
    }

    pub fn r_mut(&mut self) -> Element<ConcreteElementContainerRefMut<'_, Container::E>> {
        Element(ConcreteElementContainerRefMut {
            elem: self.imp_mut(),
        })
    }

    pub fn view_mut(&mut self) -> ElementViewMut<'_, <Container::E as ElementImpl>::Space> {
        self.imp_mut().view_mut()
    }

    pub fn axpy_inplace(
        &mut self,
        alpha: <Container::E as ElementImpl>::F,
        other: Element<impl ElementContainer<E = Container::E>>,
    ) {
        self.imp_mut().axpy_inplace(alpha, other.imp());
    }

    pub fn sum_inplace(&mut self, other: Element<impl ElementContainer<E = Container::E>>) {
        self.imp_mut().sum_inplace(other.imp());
    }

    pub fn sub_inplace(&mut self, other: Element<impl ElementContainer<E = Container::E>>) {
        self.imp_mut().sub_inplace(other.imp());
    }

    pub fn fill_inplace(&mut self, other: Element<impl ElementContainer<E = Container::E>>) {
        self.imp_mut().fill_inplace(other.imp());
    }

    pub fn scale_inplace(&mut self, alpha: <Container::E as ElementImpl>::F) {
        self.imp_mut().scale_inplace(alpha);
    }
}

impl<ElemImpl: ElementImpl> Element<ConcreteElementContainer<ElemImpl>> {
    pub fn new(elem: ElemImpl) -> Self {
        Self(ConcreteElementContainer(elem))
    }
}

impl<'a, ElemImpl: ElementImpl> Element<ConcreteElementContainerRef<'a, ElemImpl>> {
    pub fn new(elem: &'a ElemImpl) -> Self {
        Self(ConcreteElementContainerRef { elem })
    }
}

impl<'a, ElemImpl: ElementImpl> Element<ConcreteElementContainerRefMut<'a, ElemImpl>> {
    pub fn new(elem: &'a mut ElemImpl) -> Self {
        Self(ConcreteElementContainerRefMut { elem })
    }
}

// Arithmetic operations

impl<E: ElementImpl, Container1: ElementContainer<E = E>, Container2: ElementContainer<E = E>>
    std::ops::Add<Element<Container2>> for Element<Container1>
{
    type Output = ElementType<E>;

    fn add(self, other: Element<Container2>) -> Self::Output {
        let mut x = self.duplicate();
        x.sum_inplace(other);
        x
    }
}

impl<E: ElementImpl, Container1: ElementContainer<E = E>, Container2: ElementContainer<E = E>>
    std::ops::Sub<Element<Container2>> for Element<Container1>
{
    type Output = ElementType<E>;

    fn sub(self, other: Element<Container2>) -> Self::Output {
        let mut x = self.duplicate();
        x.sub_inplace(other);
        x
    }
}

impl<
        E: ElementImpl,
        Container1: ElementContainerMut<E = E>,
        Container2: ElementContainer<E = E>,
    > std::ops::AddAssign<Element<Container2>> for Element<Container1>
{
    fn add_assign(&mut self, rhs: Element<Container2>) {
        self.sum_inplace(rhs);
    }
}

impl<E: ElementImpl, Container: ElementContainer<E = E>> std::ops::Neg for Element<Container> {
    type Output = ElementType<E>;

    fn neg(self) -> Self::Output {
        let mut x = self.duplicate();
        x.scale_inplace(-<E as ElementImpl>::F::one());
        x
    }
}

impl<E: ElementImpl, Container: ElementContainer<E = E>> std::ops::Mul<<E as ElementImpl>::F>
    for Element<Container>
{
    type Output = ElementType<E>;

    fn mul(self, rhs: <E as ElementImpl>::F) -> Self::Output {
        let mut x = self.duplicate();
        x.scale_inplace(rhs);
        x
    }
}

impl<E: ElementImpl, Container: ElementContainerMut<E = E>>
    std::ops::MulAssign<<E as ElementImpl>::F> for Element<Container>
{
    fn mul_assign(&mut self, rhs: <E as ElementImpl>::F) {
        self.scale_inplace(rhs);
    }
}

impl<E: ElementImpl, Container: ElementContainer<E = E>> std::ops::Div<<E as ElementImpl>::F>
    for Element<Container>
{
    type Output = ElementType<E>;

    fn div(self, rhs: <E as ElementImpl>::F) -> Self::Output {
        let mut x = self.duplicate();
        x.scale_inplace(<E as ElementImpl>::F::one() / rhs);
        x
    }
}

impl<E: ElementImpl, Container: ElementContainerMut<E = E>>
    std::ops::DivAssign<<E as ElementImpl>::F> for Element<Container>
{
    fn div_assign(&mut self, rhs: <E as ElementImpl>::F) {
        self.scale_inplace(<E as ElementImpl>::F::one() / rhs);
    }
}

impl<
        E: ElementImpl,
        Container1: ElementContainerMut<E = E>,
        Container2: ElementContainer<E = E>,
    > std::ops::SubAssign<Element<Container2>> for Element<Container1>
{
    fn sub_assign(&mut self, rhs: Element<Container2>) {
        *self += -rhs;
    }
}

macro_rules! impl_element_scalar_mul {
    ($scalar:ty) => {
        impl<E: ElementImpl<F = $scalar>, Container: ElementContainer<E = E>>
            std::ops::Mul<Element<Container>> for $scalar
        {
            type Output = ElementType<E>;

            fn mul(self, rhs: Element<Container>) -> Self::Output {
                rhs * self
            }
        }
    };
}

impl_element_scalar_mul!(f32);
impl_element_scalar_mul!(f64);
impl_element_scalar_mul!(c32);
impl_element_scalar_mul!(c64);

pub trait ScalarTimesElement<Container: ElementContainer>:
    std::ops::Mul<Element<Container>>
{
}

impl<T: RlstScalar, E: ElementImpl<F = T>, Container: ElementContainer<E = E>>
    ScalarTimesElement<Container> for T
where
    T: std::ops::Mul<Element<Container>>,
{
}

pub type ElementType<ElemImpl> = Element<ConcreteElementContainer<ElemImpl>>;
pub type ElementTypeRef<'a, ElemImpl> = Element<ConcreteElementContainerRef<'a, ElemImpl>>;
pub type ElementTypeRefMut<'a, ElemImpl> = Element<ConcreteElementContainerRefMut<'a, ElemImpl>>;
