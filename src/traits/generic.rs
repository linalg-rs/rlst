//! Generic traits.

/// Return a view onto an object.
pub trait View {
    type Impl;
    /// Get a view onto an object.
    fn view(&self) -> &Self::Impl;
}

/// Return a mutable view onto an object.
pub trait ViewMut: View {
    /// Get a mutable view onto an object.
    fn view_mut(&mut self) -> &Self::Impl;
}
