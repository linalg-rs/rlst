//! An indexable vector is the standard type for n-dimensional containers

use crate::traits::index_layout::IndexLayout;
use rlst_common::types::{IndexType, Scalar};

pub trait IndexableVector {
    type Item: Scalar;
    type Ind: IndexLayout;

    type View<'a>: IndexableVectorView
    where
        Self: 'a;
    type ViewMut<'a>: IndexableVectorView
    where
        Self: 'a;

    fn view<'a>(&'a self) -> Option<Self::View<'a>>;
    fn view_mut<'a>(&'a mut self) -> Option<Self::ViewMut<'a>>;

    fn index_layout(&self) -> &Self::Ind;
}

pub trait IndexableVectorView {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = &'a Self::T>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;

    fn get(&self, index: IndexType) -> Option<&Self::T>;

    unsafe fn get_unchecked(&self, index: IndexType) -> &Self::T;

    fn len(&self) -> IndexType;

    fn data(&self) -> &[Self::T];
}

pub trait IndexableVectorViewMut: IndexableVectorView {
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::T>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn get_mut(&mut self, index: IndexType) -> Option<&mut Self::T>;

    unsafe fn get_unchecked_mut(&mut self, index: IndexType) -> &mut Self::T;

    fn data_mut(&mut self) -> &mut [Self::T];
}

/// Inner product with another object.
pub trait Inner: IndexableVector {
    fn inner(&self, other: &Self) -> rlst_common::types::SparseLinAlgResult<Self::Item>;
}

/// Take the sum of the squares of the absolute values of the entries.
pub trait AbsSquareSum: IndexableVector {
    fn abs_square_sum(&self) -> <Self::Item as Scalar>::Real;
}

/// Return the 1-Norm (Sum of absolute values of the entries).
pub trait Norm1: IndexableVector {
    fn norm_1(&self) -> <Self::Item as Scalar>::Real;
}

/// Return the 2-Norm (Sqrt of the sum of squares).
pub trait Norm2: IndexableVector {
    fn norm_2(&self) -> <Self::Item as Scalar>::Real;
}

/// Return the supremum norm (largest absolute value of the entries).
pub trait NormInfty: IndexableVector {
    fn norm_infty(&self) -> <Self::Item as Scalar>::Real;
}

/// Swap entries with another vector.
pub trait Swap: IndexableVector {
    fn swap(&mut self, other: &mut Self) -> rlst_common::types::SparseLinAlgResult<()>;
}

/// Fill vector by copying from another vector.
pub trait Fill: IndexableVector {
    fn fill(&mut self, other: &Self) -> rlst_common::types::SparseLinAlgResult<()>;
}

/// Multiply entries with a scalar.
pub trait ScalarMult: IndexableVector {
    fn scalar_mult(&mut self, scalar: Self::Item);
}

/// Compute self -> alpha * other + self.
pub trait MultSumInto: IndexableVector {
    fn mult_sum_into(
        &mut self,
        other: &Self,
        scalar: Self::Item,
    ) -> rlst_common::types::SparseLinAlgResult<()>;
}
