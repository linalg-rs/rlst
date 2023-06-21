//! Trait for QR Decomposition
use super::types::*;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{MatrixD, RandomAccessByValue, Shape, Stride};

pub trait QRTrait {
    type T: Scalar;
    type Q;
    type R;

    fn q(&self, mode: QrMode) -> RlstResult<Self::Q>;

    fn r(&self) -> RlstResult<Self::R>;

    fn permutation(&self) -> &Vec<usize>;
}

pub trait QRDecomposableTrait {
    type T: Scalar;
    type Out;

    fn qr(self, pivoting: PivotMode) -> RlstResult<Self::Out>;
    fn solve_least_squares<Rhs: RandomAccessByValue<Item = Self::T> + Shape + Stride>(
        self,
        rhs: &Rhs,
        trans: TransposeMode,
    ) -> RlstResult<MatrixD<Self::T>>;
}
