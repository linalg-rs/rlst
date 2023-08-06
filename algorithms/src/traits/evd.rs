//! Trait for SVD
use rlst_common::types::{RlstResult, Scalar};
use rlst_dense::MatrixD;

#[derive(PartialEq)]
pub enum EigenvectorMode {
    Left,
    Right,
    All,
    None,
}

pub trait SymEvd {
    type T: Scalar;

    #[allow(clippy::type_complexity)]
    fn sym_evd(
        self,
        mode: EigenvectorMode,
    ) -> RlstResult<(
        Vec<<Self::T as Scalar>::Real>,
        Option<MatrixD<Self::T>>,
        Option<MatrixD<Self::T>>,
    )>;
}

pub trait Evd: Sized {
    type T: Scalar;

    #[allow(clippy::type_complexity)]
    fn evd(
        self,
        mode: EigenvectorMode,
    ) -> RlstResult<(
        Vec<<Self::T as Scalar>::Complex>,
        Option<MatrixD<<Self::T as Scalar>::Complex>>,
        Option<MatrixD<<Self::T as Scalar>::Complex>>,
    )>;

    fn eigenvalues(self) -> RlstResult<Vec<<Self::T as Scalar>::Complex>> {
        let (eigvals, _, _) = self.evd(EigenvectorMode::None)?;
        Ok(eigvals)
    }
}
