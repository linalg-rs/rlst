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

    fn view(&self) -> Self::View<'_>;

    fn view_mut(&mut self) -> Self::ViewMut<'_>;
}

// The view type associated with elements of linear spaces.
pub type ElementView<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::View<'a>;

// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::ViewMut<'a>;
