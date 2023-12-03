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
