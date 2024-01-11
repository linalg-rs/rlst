use super::ElementView;
use super::LinearSpace;
use rlst_common::types::Scalar;

pub trait NormedSpace: LinearSpace {
    /// Norm of a vector.
    fn norm<'a>(&'a self, x: &ElementView<'a, Self>) -> <Self::F as Scalar>::Real;
}
