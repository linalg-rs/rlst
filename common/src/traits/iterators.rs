//! Iterator traits

use crate::types::Scalar;

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator {
    type Item: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::Item)>
    where
        Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_>;
}

/// Iterate through the elements in column-major ordering.
pub trait ColumnMajorIterator {
    type Item: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::Item>
    where
        Self: 'a;

    fn iter_col_major(&self) -> Self::Iter<'_>;
}

/// Default iterator.
pub trait DefaultIterator {
    type Item: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::Item>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;
}

/// Mutable iterator through the elements in column-major ordering.
pub trait ColumnMajorIteratorMut {
    type Item: Scalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    fn iter_col_major_mut(&mut self) -> Self::IterMut<'_>;
}

/// Mutable default iterator.
pub trait DefaultIteratorMut {
    type Item: Scalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

/// Iterate through the diagonal.
pub trait DiagonalIterator {
    type Item: Scalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::Item>
    where
        Self: 'a;

    fn iter_diag(&self) -> Self::Iter<'_>;
}

/// Mutable iterator through the diagonal.
pub trait DiagonalIteratorMut {
    type Item: Scalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    fn iter_diag_mut(&mut self) -> Self::IterMut<'_>;
}

/// Apply an `FnMut` closure to each element.
pub trait ForEach {
    type Item: Scalar;

    fn for_each<F: FnMut(&mut Self::Item)>(&mut self, f: F);
}
