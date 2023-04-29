//! Traits for the creation of new entities.

/// Create a new object from `self` initialized to zero.
pub trait NewFromSelf {
    type Out;
    fn new_from_self(&self) -> Self::Out;
}
