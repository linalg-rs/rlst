//! Trait for Pseudo Inverse
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::MatrixD;

pub trait Pinv {
    type T: Scalar;

    #[allow(clippy::type_complexity)]
    fn pinv(
        self,
    ) -> RlstResult<(
        Vec<<Self::T as Scalar>::Real>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )>;
}
