//! Useful library tools.

use crate::types::*;
use rand::prelude::*;
use rand_distr::Distribution;

/// This trait implements a simple convenient function to return random scalars
/// from a given random number generator and distribution. For complex types the
/// generator and distribution are separately applied to obtain the real and imaginary
/// part of the random number.
pub trait RandScalar: HScalar {
    /// Returns a random number from a given random number generator `rng` and associated
    /// distribution `dist`.
    fn random_scalar<R: Rng, D: Distribution<<Self as HScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self;
}

impl RandScalar for f32 {
    fn random_scalar<R: Rng, D: Distribution<Self>>(rng: &mut R, dist: &D) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for f64 {
    fn random_scalar<R: Rng, D: Distribution<Self>>(rng: &mut R, dist: &D) -> Self {
        dist.sample(rng)
    }
}

impl RandScalar for c32 {
    fn random_scalar<R: Rng, D: Distribution<<Self as HScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        c32::new(dist.sample(rng), dist.sample(rng))
    }
}

impl RandScalar for c64 {
    fn random_scalar<R: Rng, D: Distribution<<Self as HScalar>::Real>>(
        rng: &mut R,
        dist: &D,
    ) -> Self {
        c64::new(dist.sample(rng), dist.sample(rng))
    }
}
