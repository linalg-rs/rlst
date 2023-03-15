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

    fn view<'b>(&'b self) -> Self::View<'b>;

    fn view_mut<'b>(&'b mut self) -> Self::ViewMut<'b>;
}

// The view type associated with elements of linear spaces.
pub type ElementView<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::View<'a>;

// The mutable view type associated with elements of linear spaces.
pub type ElementViewMut<'a, Space> = <<Space as LinearSpace>::E<'a> as Element>::ViewMut<'a>;
