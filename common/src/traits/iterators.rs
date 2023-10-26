//! Iterator traits

use crate::types::Scalar;

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::T)>
    where
        Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_>;
}

/// Iterate through the elements in column-major ordering.
pub trait ColumnMajorIterator {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::T>
    where
        Self: 'a;

    fn iter_col_major(&self) -> Self::Iter<'_>;
}

/// Mutable iterator through the elements in column-major ordering.
pub trait ColumnMajorIteratorMut {
    type T: Scalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::T>
    where
        Self: 'a;

    fn iter_col_major_mut(&mut self) -> Self::IterMut<'_>;
}

/// Iterate through the diagonal.
pub trait DiagonalIterator {
    type T: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::T>
    where
        Self: 'a;

    fn iter_diag(&self) -> Self::Iter<'_>;
}

/// Mutable iterator through the diagonal.
pub trait DiagonalIteratorMut {
    type T: Scalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::T>
    where
        Self: 'a;

    fn iter_diag_mut(&mut self) -> Self::IterMut<'_>;
}

/// Apply an `FnMut` closure to each element.
pub trait ForEach {
    type T: Scalar;

    fn for_each<F: FnMut(&mut Self::T)>(&mut self, f: F);
}
