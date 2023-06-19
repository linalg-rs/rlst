//! Trait for QR Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{MatrixD, RandomAccessByValue, RawAccessMut, Shape, Stride};

/// QR Mode
///
/// Full: Return the full Q matrix.
/// Reduced: Return the reduced Q matrix.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub enum Mode {
    Full,
    Reduced,
}

pub trait QRTrait {
    type T: Scalar;
    type Q;
    type R;

    fn q(&self, mode: Mode) -> RlstResult<Self::Q>;

    fn r(&self) -> RlstResult<Self::R>;
}

pub trait QRDecomposableTrait {
    type T: Scalar;
    type Out;

    fn qr(self) -> RlstResult<Self::Out>;
    fn solve_least_squares<Rhs: RandomAccessByValue<Item = Self::T> + Shape + Stride>(
        self,
        rhs: &Rhs,
        trans: TransposeMode,
    ) -> RlstResult<MatrixD<Self::T>>;
}
