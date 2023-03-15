//! Traits for sparse matrices

pub use crate::linalg::indexable_vector::*;
pub use crate::linalg::matrix::Matrix;
pub use crate::types::{IndexType, Scalar};

/// Compute the Frobenious norm of a matrix.
pub trait NormFrob: Matrix {
    fn norm_frob(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the 1-Norm of a matrix.
pub trait Norm1: Matrix {
    fn norm_1(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the 2-Norm of a matrix.
pub trait Norm2: Matrix {
    fn norm_2(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the supremum norm of a matrix.
pub trait NormInfty: Matrix {
    fn norm_infty(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the trace of a matrix.
pub trait Trace: Matrix {
    fn trace(&self) -> Self::T;
}

/// Compute the real part of a matrix.
pub trait Real: Matrix {
    fn real(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the imaginary part of a matrix.
pub trait Imag: Matrix {
    fn imag(&self) -> <Self::T as Scalar>::Real;
}

/// Swap entries with another matrix.
pub trait Swap: Matrix {
    fn swap(&mut self, other: &mut Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Fill matrix by copying from another matrix.
pub trait Fill: Matrix {
    fn fill(&mut self, other: &Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Multiply entries with a scalar.
pub trait ScalarMult: Matrix {
    fn scalar_mult(&mut self, scalar: Self::T);
}

/// Compute self -> alpha * other + self.
pub trait MultSumInto: Matrix {
    fn mult_sum_into(&mut self, other: &Self, scalar: Self::T) -> crate::types::SparseLinAlgResult<()>;
}
