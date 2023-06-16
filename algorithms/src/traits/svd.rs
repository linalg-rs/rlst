//! Trait for SVD
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::MatrixD;

pub enum Mode {
    All,
    Slim,
    None,
}

pub trait Svd {
    type T: Scalar;

    #[allow(clippy::type_complexity)]
    fn svd(
        self,
        u_mode: Mode,
        vt_mode: Mode,
    ) -> RlstResult<(
        Vec<<Self::T as Scalar>::Real>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )>;
}
