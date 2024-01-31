use crate::ElementType;
use crate::InnerProductSpace;

use super::LinearSpace;
use rlst_common::types::Scalar;

pub trait NormedSpace: LinearSpace {
    /// Norm of a vector.
    fn norm<'a>(&'a self, x: &ElementType<'a, Self>) -> <Self::F as Scalar>::Real;
}

impl<S: InnerProductSpace> NormedSpace for S {
    fn norm<'a>(&'a self, x: &ElementType<'a, Self>) -> <Self::F as Scalar>::Real {
        let abs_square = self.inner(x, x).abs();
        abs_square.sqrt()
    }
}
