use rlst_common::types::Scalar;

use super::LinearSpace;

/// Elements of linear spaces.
pub trait Element {
    /// Item type of the vector.

    type Space: LinearSpace<F = Self::F, E = Self>;

    type F: Scalar;

    type View<'b>
    where
        Self: 'b;
    type ViewMut<'b>
    where
        Self: 'b;

    /// Get a view onto the element.
    fn view(&self) -> Self::View<'_>;

    /// Get a mutable view onto the element.
    fn view_mut(&mut self) -> Self::ViewMut<'_>;

    /// `self += alpha * other`.
    fn sum_into(&mut self, alpha: Self::F, other: &Self);

    /// self = other.
    fn fill_from(&mut self, other: &Self);

    /// `self *= alpha`.
    fn scale_in_place(&mut self, alpha: Self::F);
}

// The view type associated with elements of linear spaces.
pub type ElementView<'view, Space> = <<Space as LinearSpace>::E as Element>::View<'view>;

// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'view, Space> = <<Space as LinearSpace>::E as Element>::ViewMut<'view>;
