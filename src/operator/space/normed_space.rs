//! Normed spaces
use crate::operator::ElementImplType;
use crate::operator::InnerProductSpace;

use super::LinearSpace;
use crate::dense::types::RlstScalar;

impl<S: InnerProductSpace> NormedSpace for S {
    fn norm(&self, x: &ElementImplType<Self>) -> <Self::F as RlstScalar>::Real {
        let abs_square = self.inner_product(x, x).abs();
        abs_square.sqrt()
    }
}
