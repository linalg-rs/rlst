use super::LinearSpace;

/// Elements of linear spaces.
pub trait Element {
    /// Item type of the vector.
    type Space: LinearSpace;

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
}

// The view type associated with elements of linear spaces.
pub type ElementView<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::View<'a>;

// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::ViewMut<'a>;
