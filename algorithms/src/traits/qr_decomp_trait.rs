//! Trait for QR Decomposition
use crate::lapack::TransposeMode;
pub use rlst_common::types::{RlstError, RlstResult, Scalar};
use rlst_dense::{RawAccessMut, Shape, Stride, MatrixD};

pub trait QRDecompTrait {
    type T: Scalar;

    fn data(&self) -> &[Self::T];

    fn shape(&self) -> (usize, usize);

    fn stride(&self) -> (usize, usize);

    fn q_mult<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<Rhs>;

    fn get_q(&self) -> RlstResult<MatrixD<Self::T>>;

    fn get_r(&self) -> RlstResult<MatrixD<Self::T>>;

    fn solve_qr<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        &self,
        rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<Rhs>;
}

pub trait QR {
    type T: Scalar;
    type Out;

    fn qr(self) -> RlstResult<Self::Out>;
    fn qr_and_solve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
        self,
        rhs: Rhs,
        trans: TransposeMode,
    ) -> RlstResult<(Rhs)>;

    // fn qr_col_pivot(self) -> RlstResult<Self::Out>;
    // fn qr_col_pivot_free_cols(self,jpvt: Vec<i32>) -> RlstResult<Self::Out>;
    // fn qr_col_pivot_and_solve<Rhs: RawAccessMut<T = Self::T> + Shape + Stride>(
    //     self,
    //     rhs: Rhs,
    //     trans: TransposeMode,
    // ) -> RlstResult<()>;
}