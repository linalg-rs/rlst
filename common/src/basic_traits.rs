pub use crate::types::{IndexType, Scalar};

pub trait Apply<Domain> {
    type T: Scalar;
    type Range;

    /// Compute y -> alpha A x + beta y
    fn apply(&self, alpha: Self::T, x: &Domain, y: &mut Self::Range, beta: Self::T);
}

pub trait Inner {
    type T: Scalar;

    fn inner(&self, other: &Self) -> Self::T;
}
