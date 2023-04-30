//! Methods for the creation of random matrices.

use crate::data_container::DataContainerMut;
use crate::traits::*;
use rand::prelude::*;
use rand_distr::Standard;
use rand_distr::StandardNormal;
use rlst_common::tools::*;
use rlst_common::traits::*;

use super::GenericBaseMatrix;

impl<
        Item: Scalar + RandScalar,
        RS: SizeIdentifier,
        CS: SizeIdentifier,
        Data: DataContainerMut<Item = Item>,
    > GenericBaseMatrix<Item, Data, RS, CS>
where
    StandardNormal: Distribution<<Item as Scalar>::Real>,
    Standard: Distribution<<Item as Scalar>::Real>,
{
    /// Fill a matrix with normally distributed random numbers.
    pub fn fill_from_standard_normal<R: Rng>(&mut self, rng: &mut R) {
        let dist = StandardNormal;
        self.for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }

    /// Fill a matrix with equally distributed random numbers.
    pub fn fill_from_equally_distributed<R: Rng>(&mut self, rng: &mut R) {
        let dist = Standard;
        self.for_each(|val| *val = <Item>::random_scalar(rng, &dist));
    }
}
