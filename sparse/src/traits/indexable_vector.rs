//! An indexable vector is the standard type for n-dimensional containers

use crate::traits::index_layout::IndexLayout;
use rlst_common::types::RlstScalar;

pub trait IndexableVector {
    type Item: RlstScalar;
    type Ind: IndexLayout;

    type View<'a>: IndexableVectorView
    where
        Self: 'a;
    type ViewMut<'a>: IndexableVectorView
    where
        Self: 'a;

    fn view(&self) -> Option<Self::View<'_>>;
    fn view_mut(&mut self) -> Option<Self::ViewMut<'_>>;

    fn index_layout(&self) -> &Self::Ind;
}

pub trait IndexableVectorView {
    type T: RlstScalar;
    type Iter<'a>: std::iter::Iterator<Item = &'a Self::T>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;

    fn get(&self, index: usize) -> Option<&Self::T>;

    /// # Safety
    /// `index` must not exceed bounds.
    unsafe fn get_unchecked(&self, index: usize) -> &Self::T;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn data(&self) -> &[Self::T];
}

pub trait IndexableVectorViewMut: IndexableVectorView {
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::T>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;

    fn get_mut(&mut self, index: usize) -> Option<&mut Self::T>;

    /// # Safety
    /// `index` must not exceed bounds.
    unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut Self::T;

    fn data_mut(&mut self) -> &mut [Self::T];
}

/// Inner product with another object.
pub trait Inner: IndexableVector {
    fn inner(&self, other: &Self) -> rlst_common::types::RlstResult<Self::Item>;
}

/// Take the sum of the squares of the absolute values of the entries.
pub trait AbsSquareSum: IndexableVector {
    fn abs_square_sum(&self) -> <Self::Item as RlstScalar>::Real;
}

/// Return the 1-Norm (Sum of absolute values of the entries).
pub trait Norm1: IndexableVector {
    fn norm_1(&self) -> <Self::Item as RlstScalar>::Real;
}

/// Return the 2-Norm (Sqrt of the sum of squares).
pub trait Norm2: IndexableVector {
    fn norm_2(&self) -> <Self::Item as RlstScalar>::Real;
}

/// Return the supremum norm (largest absolute value of the entries).
pub trait NormInfty: IndexableVector {
    fn norm_infty(&self) -> <Self::Item as RlstScalar>::Real;
}

/// Swap entries with another vector.
pub trait Swap: IndexableVector {
    fn swap(&mut self, other: &mut Self) -> rlst_common::types::RlstResult<()>;
}

/// Fill vector by copying from another vector.
pub trait Fill: IndexableVector {
    fn fill(&mut self, other: &Self) -> rlst_common::types::RlstResult<()>;
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
    ) -> rlst_common::types::RlstResult<()>;
}
