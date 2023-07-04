//! Trait for Pseudo Inverse
use num::Float;

use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::MatrixD;

pub trait Pinv {
    type T: Scalar + Float;

    #[allow(clippy::type_complexity)]
    fn pinv(
        self,
        threshold: Option<Self::T>,
    ) -> RlstResult<(
        Option<Vec<<Self::T as Scalar>::Real>>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )>;
}
