//! Marker traits for number relations expressed through types.

/// Is one more than
pub trait IsGreaterByOne<const N: usize> {}
/// Is one less than
pub trait IsSmallerByOne<const N: usize> {}
/// Is greater than zero
pub trait IsGreaterZero {}

/// Is smaller than a given number a
pub trait IsSmallerThan<const N: usize> {}
