//! Methods for the creation of random matrices.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Standard;
use rand_distr::StandardNormal;

use crate::traits::iterators::ArrayIteratorMut;
use crate::traits::rlst_num::RandScalar;
use crate::traits::rlst_num::RlstScalar;

use super::Array;

impl<Item, ArrayImpl, const NDIM: usize> Array<ArrayImpl, NDIM>
where
    Item: RlstScalar + RandScalar,
    Self: ArrayIteratorMut<Item = Item>,
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<<Item as RlstScalar>::Real>,
{
    /// Fill an array with normally distributed random numbers.
    pub fn fill_from_standard_normal<R: Rng>(&mut self, rng: &mut R) {
        let dist = StandardNormal;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }

    /// Fill an array with equally distributed random numbers.
    pub fn fill_from_equally_distributed<R: Rng>(&mut self, rng: &mut R) {
        let dist = Standard;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }

    /// Fill with equally distributed numbers using a given `seed`.
    pub fn fill_from_seed_equally_distributed(&mut self, seed: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let dist = Standard;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(&mut rng, &dist));
    }

    /// Fill with normally distributed numbers using a given seed.
    pub fn fill_from_seed_normally_distributed(&mut self, seed: usize) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let dist = StandardNormal;
        self.iter_mut()
            .for_each(|val| *val = <Item>::random_scalar(&mut rng, &dist));
    }
}
