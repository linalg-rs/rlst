//! Dense matrix traits

pub mod accessors;

pub use crate::dense::linalg::{
    inverse::MatrixInverse, lu::MatrixLuDecomposition, pseudo_inverse::MatrixPseudoInverse,
    qr::MatrixQrDecomposition, svd::MatrixSvd,
};
pub use accessors::*;

use crate::dense::types::RlstScalar;
use crate::dense::types::TransMode;

/// Shape of an object
pub trait Shape<const NDIM: usize> {
    /// Return the shape of the object.
    fn shape(&self) -> [usize; NDIM];

    /// Return true if a dimension is 0.
    fn is_empty(&self) -> bool {
        let shape = self.shape();
        for elem in shape {
            if elem == 0 {
                return true;
            }
        }
        false
    }
}

/// Stride of an object
pub trait Stride<const NDIM: usize> {
    /// Return the stride of the object.
    fn stride(&self) -> [usize; NDIM];
}

/// Number of elements
pub trait NumberOfElements {
    /// Return the number of elements.
    fn number_of_elements(&self) -> usize;
}

/// Resize in place
pub trait ResizeInPlace<const NDIM: usize> {
    /// Resize an operator in place
    fn resize_in_place(&mut self, shape: [usize; NDIM]);
}

/// Multiply into
pub trait MultInto<First, Second> {
    /// Item type
    type Item: RlstScalar;
    /// Multiply First * Second and sum into Self
    fn simple_mult_into(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
    {
        self.mult_into(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into
    fn mult_into(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

/// Multiply into with resize
pub trait MultIntoResize<First, Second> {
    /// Item type
    type Item: RlstScalar;
    /// Multiply First * Second and sum into Self. Allow to resize Self if necessary
    fn simple_mult_into_resize(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
    {
        self.mult_into_resize(
            TransMode::NoTrans,
            TransMode::NoTrans,
            <Self::Item as num::One>::one(),
            arr_a,
            arr_b,
            <Self::Item as num::Zero>::zero(),
        )
    }
    /// Multiply into with resize
    fn mult_into_resize(
        self,
        transa: TransMode,
        transb: TransMode,
        alpha: Self::Item,
        arr_a: First,
        arr_b: Second,
        beta: Self::Item,
    ) -> Self;
}

/// Default iterator.
pub trait DefaultIterator {
    /// Item type
    type Item: RlstScalar;
    /// Iterator type
    type Iter<'a>: std::iter::Iterator<Item = Self::Item>
    where
        Self: 'a;
    /// Get iterator
    fn iter(&self) -> Self::Iter<'_>;
}

/// Mutable default iterator.
pub trait DefaultIteratorMut {
    /// Item type
    type Item: RlstScalar;
    /// Iterator
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;
    /// Get mutable iterator
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator {
    /// Item type
    type Item: RlstScalar;
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
    ) -> crate::dense::array::iterators::MultiIndexIterator<T, I, NDIM>;
}
