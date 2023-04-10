//! Traits for sparse matrices

pub use crate::linalg::indexable_matrix::IndexableMatrix;
pub use crate::linalg::indexable_vector::*;
pub use crate::types::{usize, Scalar};

/// Compute the Frobenious norm of a matrix.
pub trait NormFrob: IndexableMatrix {
    fn norm_frob(&self) -> <Self::Item as Scalar>::Real;
}

/// Compute the 1-Norm of a matrix.
pub trait Norm1: IndexableMatrix {
    fn norm_1(&self) -> <Self::Item as Scalar>::Real;
}

/// Compute the 2-Norm of a matrix.
pub trait Norm2: IndexableMatrix {
    fn norm_2(&self) -> <Self::Item as Scalar>::Real;
}

/// Compute the supremum norm of a matrix.
pub trait NormInfty: IndexableMatrix {
    fn norm_infty(&self) -> <Self::Item as Scalar>::Real;
}

/// Compute the trace of a matrix.
pub trait Trace: IndexableMatrix {
    fn trace(&self) -> Self::Item;
}

/// Compute the real part of a matrix.
pub trait Real: IndexableMatrix {
    fn real(&self) -> <Self::Item as Scalar>::Real;
}

/// Compute the imaginary part of a matrix.
pub trait Imag: IndexableMatrix {
    fn imag(&self) -> <Self::Item as Scalar>::Real;
}

/// Swap entries with another matrix.
pub trait Swap: IndexableMatrix {
    fn swap(&mut self, other: &mut Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Fill matrix by copying from another matrix.
pub trait Fill: IndexableMatrix {
    fn fill(&mut self, other: &Self) -> crate::types::SparseLinAlgResult<()>;
}

/// Multiply entries with a scalar.
pub trait ScalarMult: IndexableMatrix {
    fn scalar_mult(&mut self, scalar: Self::Item);
}

/// Compute self -> alpha * other + self.
pub trait MultSumInto: IndexableMatrix {
    fn mult_sum_into(
        &mut self,
        other: &Self,
        scalar: Self::Item,
    ) -> crate::types::SparseLinAlgResult<()>;
}

/// Compute y-> alpha A x + y
pub trait MatMul: IndexableMatrix {
    type Vec: IndexableVector<Item = Self::Item>;

    fn matmul(&self, alpha: Self::Item, x: &Self::Vec, beta: Self::Item, y: &mut Self::Vec);
}
