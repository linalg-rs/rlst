use super::{
    accessors::{RandomAccessByValue, RandomAccessMut},
    array::BaseItem,
};

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator: BaseItem {
    /// Iterator
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::Item)>
    where
        Self: 'a;
    /// Get iterator
    fn iter_aij(&self) -> Self::Iter<'_>;
}

/// Helper trait that returns from an enumeration iterator a new iterator
/// that converts the 1d index into a multi-index.
pub trait AsMultiIndex<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> {
    /// Get multi-index
    fn multi_index(
        self,
        shape: [usize; NDIM],
    ) -> crate::dense::array::iterators::MultiIndexIterator<I, NDIM>;
}

/// Provides a default iterator over elements of an array.
///
/// The returned iterator is expected to iterate through the array
/// in column major order independent of the underlying memory layout.
pub trait ArrayIterator: BaseItem {
    /// Type of the iterator.
    type Iter<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces elements of type `Self::Item`.
    fn iter(&self) -> Self::Iter<'_>;
}

/// Provides a default mutable iterator over elements of an array.
///
/// The returned iterator is expected to iterate through the array
/// in column major order independent of the underlying memory layout.
pub trait ArrayIteratorMut: BaseItem {
    /// Type of the iterator.
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces mutable references to elements of type `Self::Item`.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

/// Get an iterator to the diagonal of an array.
pub trait GetDiag: BaseItem {
    /// Type of the iterator.
    type Iter<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    /// Return an iterator for the diagonal of an array.
    fn diag_iter(&self) -> Self::Iter<'_>;
}

/// Get a mutable iterator to the diagonal of an array.
pub trait GetDiagMut: BaseItem {
    /// Type of the iterator.
    type Iter<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Return a mutable iterator for the diagonal of an array.
    fn diag_iter_mut(&mut self) -> Self::Iter<'_>;
}

/// Column Iterator.
pub trait ColumnIterator: BaseItem {
    /// Column type.
    type Col<'a>: RandomAccessByValue<1, Item = Self::Item>
    where
        Self: 'a;

    /// Type of the iterator.

    type Iter<'a>: Iterator<Item = Self::Col<'a>>
    where
        Self: 'a;

    fn col_iter(&self) -> Self::Iter<'_>;
}

/// Row Iterator.
pub trait RowIterator: BaseItem {
    /// Column type.
    type Row<'a>: RandomAccessByValue<1, Item = Self::Item>
    where
        Self: 'a;

    /// Type of the iterator.

    type Iter<'a>: Iterator<Item = Self::Col<'a>>
    where
        Self: 'a;

    fn row_iter(&self) -> Self::Iter<'_>;
}

/// Mutable column Iterator.
pub trait ColumnIteratorMut: BaseItem {
    /// Column type.
    type Col<'a>: RandomAccessMut<1, Item = Self::Item>
    where
        Self: 'a;

    /// Type of the iterator.

    type Iter<'a>: Iterator<Item = Self::Col<'a>>
    where
        Self: 'a;

    fn col_iter_mut(&mut self) -> Self::Iter<'_>;
}

/// Row Iterator.
pub trait RowIteratorMut: BaseItem {
    /// Column type.
    type Row<'a>: RandomAccessMut<1, Item = Self::Item>
    where
        Self: 'a;

    /// Type of the iterator.

    type Iter<'a>: Iterator<Item = Self::Row<'a>>
    where
        Self: 'a;

    fn row_iter_mut(&mut self) -> Self::Iter<'_>;
}
