use crate::ElementType;
use crate::InnerProductSpace;

use super::LinearSpace;
use rlst_dense::types::Scalar;

pub trait NormedSpace: LinearSpace {
    /// Norm of a vector.
    fn norm(&self, x: &ElementType<Self>) -> <Self::F as Scalar>::Real;
}

impl<S: InnerProductSpace> NormedSpace for S {
    fn norm(&self, x: &ElementType<Self>) -> <Self::F as Scalar>::Real {
        let abs_square = self.inner(x, x).abs();
        abs_square.sqrt()
    }
}
