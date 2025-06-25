//! Methods for the creation of random matrices.

use crate::dense::tools::RandScalar;
use crate::dense::traits::{
    DefaultIteratorMut, Shape, UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::RlstScalar;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Standard;
use rand_distr::StandardNormal;

use super::Array;

impl<
        Item: RlstScalar + RandScalar,
        ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
            + UnsafeRandomAccessMut<NDIM, Item = Item>
            + Shape<NDIM>,
        const NDIM: usize,
    > Array<Item, ArrayImpl, NDIM>
where
    StandardNormal: Distribution<<Item as RlstScalar>::Real>,
    Standard: Distribution<<Item as RlstScalar>::Real>,
    <Item as RlstScalar>::Real: RandScalar
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

    /// Fill an array with real equally distributed random numbers.
    pub fn fill_from_equally_distributed_real<R: Rng>(&mut self, rng: &mut R) {
        let dist = Standard;
        self.iter_mut()
            .for_each(|val| *val = Item::from_real(<<Item as RlstScalar>::Real>::random_scalar(
                rng, &dist,
            )));
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
