//! This module defines typical traits for linear algebra operations.

use crate::types::Scalar;
use crate::linalg::indexable_vector::IndexableVector;

/// Inner product with another object.
pub trait Inner: IndexableVector {
    fn inner(&self, other: &Self) -> crate::types::SparseLinAlgResult<Self::T>;
}

/// Take the sum of the squares of the absolute values of the entries.
pub trait AbsSquareSum: IndexableVector {
    fn abs_square_sum(&self) -> <Self::T as Scalar>::Real;
}

/// Return the 1-Norm (Sum of absolute values of the entries).
pub trait Norm1: IndexableVector {
    fn norm_1(&self) -> <Self::T as Scalar>::Real;
}

/// Return the 2-Norm (Sqrt of the sum of squares).
pub trait Norm2: IndexableVector {
    fn norm_2(&self) -> <Self::T as Scalar>::Real;
}

/// Return the supremum norm (largest absolute value of the entries).
pub trait NormInfty: IndexableVector {
    fn norm_infty(&self) -> <Self::T as Scalar>::Real;
}

/// Swap entries with another vector.
pub trait Swap: IndexableVector {
    fn swap(&mut self, other: &mut Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Fill vector by copying from another vector.
pub trait Fill: IndexableVector {
    fn fill(&mut self, other: &Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Multiply entries with a scalar.
pub trait ScalarMult: IndexableVector {
    fn scalar_mult(&mut self, scalar: Self::T);
}

/// Compute self -> alpha * other + self.
pub trait MultSumInto: IndexableVector {
    fn mult_sum_into(&mut self, other: &Self, scalar: Self::T) -> crate::types::SparseLinAlgResult<()>;
}
