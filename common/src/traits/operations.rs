//! Traits for operations that do not change `self`.

use crate::types::Scalar;

/// Take the sum with `other`.
pub trait Sum<Other> {
    type Out;

    fn sum(&self, other: &Other) -> Self::Out;
}

/// Subtract `other` from `self`.
pub trait Sub<Other> {
    type Out;

    fn sub(&self, other: &Other) -> Self::Out;
}

/// Compute `y -> alpha * self + beta y`.
pub trait Apply<Domain> {
    type T: Scalar;
    type Out;

    fn apply(&self, alpha: Self::T, x: &Domain, y: &mut Self::Out, beta: Self::T);
}

/// Compute the 1-norm of an object.
pub trait Norm1 {
    type T: Scalar;

    fn norm1(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the inf-norm of an object.
pub trait NormInf {
    type T: Scalar;

    fn norm_inf(&self) -> <Self::T as Scalar>::Real;
}

/// Compute the inner product of an object with `other`.
/// The convention for complex objects is that the complex-conjugate
/// of `other` is taken for the inner product.
pub trait Inner {
    type T: Scalar;

    fn inner(&self, other: &Self) -> Self::T;
}

/// Compute a dual form of an object with `other`.
/// The convention for complex objecdts is that the complex-conjugate
/// of `other` is taken when appropriate for the dual form.
pub trait Dual {
    type T: Scalar;
    type Other;

    fn dual(&self, other: &Self::Other) -> Self::T;
}

/// Compute the sum of squares of the absolute values of the entries.
pub trait SquareSum {
    type T: Scalar;

    fn square_sum(&self) -> <Self::T as Scalar>::Real;
}

/// Transpose of an operator
pub trait Transpose {
    type Out;

    fn transpose(&self) -> Self::Out;
}

/// Conjugate transpose of an operator
pub trait ConjTranspose {
    type Out;

    fn conj_transpose(&self) -> Self::Out;
}

/// Permute the columns of an operator
///
/// `permutation` is a permutation vector such
/// that if permutation[i] = k
/// then the ith column of the output matrix
/// is the kth column of the input matrix.
pub trait PermuteColumns {
    type Out;

    fn permute_columns(&self, permutation: &[usize]) -> Self::Out;
}

/// Permute the rows of an operator
///
/// `permutation` is a permutation vector such
/// that if permutation\[i\] = k
/// then the ith row of the output matrix
/// is the kth row of the input matrix.

pub trait PermuteRows {
    type Out;

    fn permute_rows(&self, permutation: &[usize]) -> Self::Out;
}
