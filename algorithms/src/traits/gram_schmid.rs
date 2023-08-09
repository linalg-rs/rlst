//! Gram Schmidt trait
use rlst_dense::types::Scalar;
use rlst_dense::MatrixD;

use crate::traits::basis::*;

pub trait GramSchmidt {
    type Basis: GrowableBasis;
    type T: Scalar;

    fn orthogonalize_element(elem: <Self::Basis as Basis>::Element);
    fn get_r() -> MatrixD<Self::T>;
}
