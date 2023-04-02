//! Basic trait for matrices

use crate::traits::index_layout::IndexLayout;
use crate::traits::indexable_vector::IndexableVector;
use rlst_common::types::{IndexType, Scalar};

pub enum DenseMatrixLayout {
    RowMajor((IndexType, IndexType)),
    ColMajor((IndexType, IndexType)),
}

pub trait IndexableMatrix {
    type Item: Scalar;
    type Ind: IndexLayout;

    type View<'a>
    where
        Self: 'a;
    type ViewMut<'a>
    where
        Self: 'a;

    fn view<'a>(&'a self) -> Option<Self::View<'a>>;
    fn view_mut<'a>(&'a mut self) -> Option<Self::ViewMut<'a>>;

    fn column_layout(&self) -> &Self::Ind;
    fn row_layout(&self) -> &Self::Ind;

    fn shape(&self) -> (IndexType, IndexType) {
        (
            self.row_layout().number_of_global_indices(),
            self.column_layout().number_of_global_indices(),
        )
    }
}

pub trait IndexableDenseMatrixView {
    type T: Scalar;

    /// Return a reference to the element at position (`row`, `col`).
    unsafe fn get_unchecked(&self, row: IndexType, col: IndexType) -> &Self::T;

    /// Return a reference to the element at position `index` in one-dimensional numbering.
    unsafe fn get1d_unchecked(&self, index: IndexType) -> &Self::T;

    /// Return a reference to the element at position (`row`, `col`).
    fn get(&self, row: usize, col: usize) -> Option<&Self::T>;

    /// Return a reference to the element at position `index` in one-dimensional numbering.
    fn get1d(&self, elem: usize) -> Option<&Self::T>;

    /// Get a raw data slice.
    fn data(&self) -> &[Self::T];
}

pub trait IndexableDenseMatrixViewMut {
    type T: Scalar;

    /// Return a mutable reference to the element at position (`row`, `col`).
    unsafe fn get_unchecked_mut(&mut self, row: IndexType, col: IndexType) -> &mut Self::T;

    /// Return a mutable reference to the element at position `index` in one-dimensional numbering.
    unsafe fn get1d_unchecked_mut(&mut self, index: IndexType) -> &mut Self::T;

    /// Return a mutable reference to the element at position (`row`, `col`).
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::T>;

    /// Return a mutable reference to the element at position `index` in one-dimensional numbering.
    fn get1d_mut(&mut self, elem: usize) -> Option<&mut Self::T>;

    /// Get a mutable raw data slice.
    fn data(&mut self) -> &mut [Self::T];
}

pub trait CsrMatrixView {
    type T: Scalar;

    fn indices(&self) -> &[IndexType];

    fn indptr(&self) -> &[IndexType];

    fn data(&self) -> &[Self::T];
}

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
    fn swap(&mut self, other: &mut Self) -> rlst_common::types::RlstResult<()>;
}

/// Fill matrix by copying from another matrix.
pub trait Fill: IndexableMatrix {
    fn fill(&mut self, other: &Self) -> rlst_common::types::RlstResult<()>;
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
    ) -> rlst_common::types::RlstResult<()>;
}

/// Compute y-> alpha A x + y
pub trait MatMul: IndexableMatrix {
    type Vec: IndexableVector<Item = Self::Item>;

    fn matmul(&self, alpha: Self::Item, x: &Self::Vec, beta: Self::Item, y: &mut Self::Vec);
}
