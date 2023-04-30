//! Interface to the various library traits.

pub use crate::types::Scalar;

pub mod accessors;
pub mod constructors;
pub mod in_place_operations;
pub mod iterators;
pub mod operations;
pub mod properties;

pub use accessors::*;
pub use constructors::*;
pub use in_place_operations::*;
pub use iterators::*;
pub use operations::*;
pub use properties::*;
