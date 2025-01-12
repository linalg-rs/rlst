//! Elements of linear spaces
use std::rc::Rc;

use crate::dense::types::RlstScalar;
use num::One;

use super::{InnerProductSpace, LinearSpace, NormedSpace};

/// An Element of a linear spaces.
pub trait Element: Clone {
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

    /// self += alpha * other.
    fn axpy(mut self, alpha: Self::F, other: &Self) -> Self
    where
        Self: Sized,
    {
        self.axpy_inplace(alpha, other);
        self
    }

    /// self += other
    fn sum(mut self, other: &Self) -> Self
    where
        Self: Sized,
    {
        self.sum_inplace(other);
        self
    }

    /// self = other
    fn fill(mut self, other: &Self) -> Self
    where
        Self: Sized,
    {
        self.fill_inplace(other);
        self
    }

    /// self = alpha * self
    fn scale(mut self, alpha: Self::F) -> Self
    where
        Self: Sized,
    {
        self.scale_inplace(alpha);
        self
    }

    /// self = -self
    fn neg(mut self) -> Self
    where
        Self: Sized,
    {
        self.neg_inplace();
        self
    }

    /// Comppute the inner product with another vector
    ///
    /// Only implemented for elements of inner product spaces.
    fn inner_product(&self, other: &Self) -> Self::F
    where
        Self::Space: super::InnerProductSpace,
    {
        self.space().inner_product(self, other)
    }

    /// Compute the norm of a vector
    ///
    /// Only implemented for elements of normed spaces.
    fn norm(&self) -> <Self::F as RlstScalar>::Real
    where
        Self::Space: super::NormedSpace,
    {
        self.space().norm(self)
    }
}

/// The view type associated with elements of linear spaces.
pub type ElementView<'view, Space> = <<Space as LinearSpace>::E as Element>::View<'view>;

/// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'view, Space> = <<Space as LinearSpace>::E as Element>::ViewMut<'view>;
