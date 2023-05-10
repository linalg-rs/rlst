//! Trait for QR Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{RawAccessMut, Shape, Stride};

pub trait QRDecompTrait {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn shape(&self) -> (usize, usize);

    fn q_x_rhs<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        rhs: &mut Rhs,
        trans: TransposeMode,
    ) -> RlstResult<()>;

    fn solve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        rhs: &mut Rhs,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}

pub trait QR {
    type T: Scalar;
    type Out;

    fn qr(self) -> RlstResult<Self::Out>;
    fn solve_qr<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        self,
        rhs: &mut Rhs,
        trans: TransposeMode,
    ) -> RlstResult<()>;
}