//! Elements of linear spaces
//!
//! An [Element] of a linear space is an abstract generalization of a vector. An element
//! is always associated with an underlying [Space] and through the space provides
//! interface to operations on elements, a norm if it is a normed space, inner products,
//! and other operations associated with the underlying space.

use crate::{Inner, InnerProductSpace, LinearSpace, Norm, NormedSpace};

use crate::base_types::{c32, c64};

/// Define a generic element type for a given implementation.
///
/// An element stores a reference to the underlying space
/// and a concrete implementation.
pub struct Element<'a, Space: LinearSpace> {
    space: &'a Space,
    imp: Space::Impl,
}

impl<'a, Space: LinearSpace> Element<'a, Space> {
    /// Create a new element from a space and an implementation.
    pub fn new(space: &'a Space, imp: <Space as LinearSpace>::Impl) -> Self {
        Self { space, imp }
    }

    /// Return a reference to the space associated with the element.
    pub fn space(&self) -> &'a Space {
        self.space
    }

    /// Return a reference to the implementation of the element.
    pub fn imp(&self) -> &Space::Impl {
        &self.imp
    }

    /// Return a mutable reference to the implementation of the element.
    pub fn imp_mut(&mut self) -> &mut Space::Impl {
        &mut self.imp
    }

    ///  Destruct into the implementation.
    pub fn into_imp(self) -> Space::Impl {
        self.imp
    }
}

//  Implementation of traits for elements.

// Addition operations

impl<'a, Space: LinearSpace> std::ops::Add<Element<'a, Space>> for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn add(self, other: Element<'a, Space>) -> Self::Output {
        self.space().add(&self, &other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Add<&Element<'a, Space>> for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn add(self, other: &Element<'a, Space>) -> Self::Output {
        self.space().add(&self, other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Add<Element<'a, Space>> for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn add(self, other: Element<'a, Space>) -> Self::Output {
        self.space().add(self, &other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Add<&Element<'a, Space>> for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn add(self, other: &Element<'a, Space>) -> Self::Output {
        self.space().add(self, other)
    }
}

// Subtraction operations

impl<'a, Space: LinearSpace> std::ops::Sub<Element<'a, Space>> for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn sub(self, other: Element<'a, Space>) -> Self::Output {
        self.space().sub(&self, &other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Sub<&Element<'a, Space>> for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn sub(self, other: &Element<'a, Space>) -> Self::Output {
        self.space().sub(&self, other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Sub<Element<'a, Space>> for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn sub(self, other: Element<'a, Space>) -> Self::Output {
        self.space().sub(self, &other)
    }
}

impl<'a, Space: LinearSpace> std::ops::Sub<&Element<'a, Space>> for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn sub(self, other: &Element<'a, Space>) -> Self::Output {
        self.space().sub(self, other)
    }
}

// Scalar multiplication operations

impl<'a, Space: LinearSpace> std::ops::Mul<Space::F> for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn mul(self, scalar: Space::F) -> Self::Output {
        self.space().scalar_mul(&scalar, &self)
    }
}

impl<'a, Space: LinearSpace> std::ops::Mul<Space::F> for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn mul(self, scalar: Space::F) -> Self::Output {
        self.space().scalar_mul(&scalar, self)
    }
}

macro_rules! impl_scalar_mult {
    ($scalar:ty) => {
        impl<'a, Space: LinearSpace<F = $scalar>> std::ops::Mul<Element<'a, Space>> for $scalar {
            type Output = Element<'a, Space>;

            fn mul(self, element: Element<'a, Space>) -> Self::Output {
                element.space().scalar_mul(&self, &element)
            }
        }

        impl<'a, Space: LinearSpace<F = $scalar>> std::ops::Mul<&Element<'a, Space>> for $scalar {
            type Output = Element<'a, Space>;

            fn mul(self, element: &Element<'a, Space>) -> Self::Output {
                element.space().scalar_mul(&self, element)
            }
        }
    };
}

impl_scalar_mult!(f64);
impl_scalar_mult!(f32);
impl_scalar_mult!(c64);
impl_scalar_mult!(c32);
impl_scalar_mult!(usize);
impl_scalar_mult!(i8);
impl_scalar_mult!(i16);
impl_scalar_mult!(i32);
impl_scalar_mult!(i64);
impl_scalar_mult!(u8);
impl_scalar_mult!(u16);
impl_scalar_mult!(u32);
impl_scalar_mult!(u64);

impl<'a, Space: LinearSpace> std::ops::Neg for Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn neg(self) -> Self::Output {
        self.space().neg(&self)
    }
}

impl<'a, Space: LinearSpace> std::ops::Neg for &Element<'a, Space> {
    type Output = Element<'a, Space>;

    fn neg(self) -> Self::Output {
        self.space().neg(self)
    }
}

impl<'a, Space: LinearSpace> std::ops::AddAssign<Element<'a, Space>> for Element<'a, Space> {
    fn add_assign(&mut self, other: Element<'a, Space>) {
        self.space().sum_inplace(self, &other);
    }
}

impl<'a, Space: LinearSpace> std::ops::AddAssign<&Element<'a, Space>> for Element<'a, Space> {
    fn add_assign(&mut self, other: &Element<'a, Space>) {
        self.space().sum_inplace(self, other);
    }
}

impl<'a, Space: LinearSpace> std::ops::SubAssign<Element<'a, Space>> for Element<'a, Space> {
    fn sub_assign(&mut self, other: Element<'a, Space>) {
        self.space().sub_inplace(self, &other);
    }
}

impl<'a, Space: LinearSpace> std::ops::SubAssign<&Element<'a, Space>> for Element<'a, Space> {
    fn sub_assign(&mut self, other: &Element<'a, Space>) {
        self.space().sub_inplace(self, other);
    }
}

impl<'a, Space: LinearSpace> std::ops::MulAssign<Space::F> for Element<'a, Space> {
    fn mul_assign(&mut self, scalar: Space::F) {
        self.space().scale_inplace(&scalar, self);
    }
}

impl<'a, Space: LinearSpace> Clone for Element<'a, Space> {
    fn clone(&self) -> Self {
        self.space().copy_from(self)
    }
}

impl<'a, Space: InnerProductSpace> Inner for Element<'a, Space> {
    type Output = Space::F;

    fn inner(&self, other: &Self) -> Self::Output {
        self.space().inner_product(self, other)
    }
}

impl<'a, Space: NormedSpace> Norm for Element<'a, Space> {
    type Output = Space::Output;

    fn norm(&self) -> Self::Output {
        self.space().norm(self)
    }
}
