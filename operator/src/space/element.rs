use crate::{FieldType, InnerProductSpace, NormedSpace};
use rlst_common::types::Scalar;

use super::LinearSpace;

/// Elements of linear spaces.
pub trait Element<'elem> {
    /// Item type of the vector.
    type Space: LinearSpace<E<'elem> = Self> + 'elem;

    type View<'b>
    where
        Self: 'b;
    type ViewMut<'b>
    where
        Self: 'b;

    /// Return the underlying space.
    fn space(&self) -> &Self::Space {
        std::unimplemented!();
    }

    /// Get a view onto the element.
    fn view(&self) -> Self::View<'_>;

    /// Get a mutable view onto the element.
    fn view_mut(&mut self) -> Self::ViewMut<'_>;

    /// `self += alpha * other`.
    fn sum_into(&mut self, alpha: <Self::Space as LinearSpace>::F, other: &Self);

    /// `self *= alpha`.
    fn scale_in_place(&mut self, alpha: <Self::Space as LinearSpace>::F);

    fn inner(&'elem self, other: &'elem Self) -> FieldType<Self::Space>
    where
        Self::Space: InnerProductSpace,
    {
        self.space().inner(self, other)
    }

    fn norm(&'elem self) -> <FieldType<Self::Space> as Scalar>::Real
    where
        Self::Space: NormedSpace,
    {
        self.space().norm(self)
    }

    fn clone(&'elem self) -> Self
    where
        Self: std::marker::Sized,
    {
        self.space().clone(self)
    }
}

// The view type associated with elements of linear spaces.
pub type ElementView<'elem, Space> =
    <<Space as LinearSpace>::E<'elem> as Element<'elem>>::View<'elem>;

// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'elem, Space> =
    <<Space as LinearSpace>::E<'elem> as Element<'elem>>::ViewMut<'elem>;
