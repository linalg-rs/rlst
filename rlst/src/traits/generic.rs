//! Generic traits.

// /// Associates  an implementation type with a trait.
// pub trait ImplTrait {
//     /// The type of the implementation.
//     type Impl;
// }

// /// Return a view onto an object.
// pub trait View: ImplTrait {
//     /// Get a view onto an object.
//     fn view(&self) -> &Self::Impl;
// }

// /// Return a mutable view onto an object.
// pub trait ViewMut: View {
//     /// Get a mutable view onto an object.
//     fn view_mut(&mut self) -> &mut Self::Impl;
// }

// /// Return the implementation value of a wrapper.
// pub trait IntoInner: ImplTrait {
//     /// Get the inner value of a struct.
//     fn into_inner(self) -> Self::Impl;
// }
