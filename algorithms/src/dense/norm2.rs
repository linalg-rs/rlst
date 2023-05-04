//! Implementation of Norm2.
use num::Zero;
use rlst_common::traits::{Copy, Scalar, SquareSum};
use rlst_dense::{
    rlst_rand_mat, DataContainer, RawAccess, Shape, SizeIdentifier, UnsafeRandomAccessByValue,
};

use crate::{
    lapack::{AsLapack, LapackCompatible, LapackData},
    traits::norm2::Norm2,
    traits::svd::{Mode, Svd},
};

impl<T: Scalar, Mat: Copy + Shape + SquareSum<T = T>> Norm2 for Mat
where
    <Mat as Copy>::Out: LapackCompatible,
    <Mat as Copy>::Out: RawAccess<T = T>,
    LapackData<T, <Mat as Copy>::Out>: Svd<T = T>,
{
    type T = <<Mat as Copy>::Out as RawAccess>::T;

    fn norm2(&self) -> <Self::T as Scalar>::Real {
        let shape = self.shape();

        if shape.0 == 1 || shape.1 == 1 {
            return self.square_sum().sqrt();
        } else {
            let (s, _, _) = self
                .copy()
                .into_lapack()
                .unwrap()
                .svd(Mode::Slim, Mode::Slim)
                .unwrap();
            s[0]
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn svd_test() {
        let mat = rlst_rand_mat![f64, (4, 3)];
        println!("The norm is {}", mat.norm2());
    }
}
