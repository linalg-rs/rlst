//! Dense matrix traits

pub mod accessors;

pub use accessors::*;

use crate::dense::types::TransMode;

/// Memory layout of an object
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    /// Column major
    ColumnMajor,
    /// Row major
    RowMajor,
    /// Unknown
    Unknown,
}

/// Shape of an object
pub trait Shape<const NDIM: usize> {
    /// Return the shape of the object.
    fn shape(&self) -> [usize; NDIM];

    /// Return true if a dimension is 0.
    fn is_empty(&self) -> bool {
        *self.shape().iter().min().unwrap() == 0
    }
}

/// Stride of an object
pub trait Stride<const NDIM: usize> {
    /// Return the stride of the object.
    fn stride(&self) -> [usize; NDIM];

    /// Return the memory layout
    fn memory_layout(&self) -> MemoryLayout {
        MemoryLayout::Unknown
    }
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
    type Item;
    /// Multiply First * Second and sum into Self
    fn simple_mult_into(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
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

/// Gemm
pub trait Gemm: Sized {
    /// Gemm
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        transa: TransMode,
        transb: TransMode,
        m: usize,
        n: usize,
        k: usize,
        alpha: Self,
        a: &[Self],
        rsa: usize,
        csa: usize,
        b: &[Self],
        rsb: usize,
        csb: usize,
        beta: Self,
        c: &mut [Self],
        rsc: usize,
        csc: usize,
    );
}

/// Multiply into with resize
pub trait MultIntoResize<First, Second> {
    /// Item type
    type Item;
    /// Multiply First * Second and sum into Self. Allow to resize Self if necessary
    fn simple_mult_into_resize(self, arr_a: First, arr_b: Second) -> Self
    where
        Self: Sized,
        Self::Item: num::One,
        Self::Item: num::Zero,
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

// /// Default iterator.
// pub trait DefaultIterator {
//     /// Item type
//     type Item;
//     /// Iterator type
//     type Iter<'a>: std::iter::Iterator<Item = Self::Item>
//     where
//         Self: 'a;
//     /// Get iterator
//     fn iter(&self) -> Self::Iter<'_>;
// }

// /// Mutable default iterator.
// pub trait DefaultIteratorMut {
//     /// Item type
//     type Item;
//     /// Iterator
//     type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
//     where
//         Self: 'a;
//     /// Get mutable iterator
//     fn iter_mut(&mut self) -> Self::IterMut<'_>;
// }

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator {
    /// Item type
    type Item;
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

/// A helper trait to implement generic operators over matrices.
pub trait AsOperatorApply {
    /// Item type
    type Item;

    /// Apply the operator to a vector.
    fn apply_extended(
        &self,
        alpha: Self::Item,
        x: &[Self::Item],
        beta: Self::Item,
        y: &mut [Self::Item],
    );
}

/// Provides a default iterator over elements of an array.
///
/// The returned iterator is expected to iterate through the array
/// in column major order independent of the underlying memory layout.
pub trait ArrayIterator {
    /// Item type of the iterator.
    type Item;
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
pub trait ArrayIteratorMut {
    /// Item type of the iterator.
    type Item;
    /// Type of the iterator.
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces mutable references to elements of type `Self::Item`.
    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

// Get an iterator to the diagonal of an array.
pub trait GetDiag {
    type Item;
    type Iter<'a>: Iterator<Item = Self::Item>
    where
        Self: 'a;

    fn diag_iter(&self) -> Self::Iter<'_>;
}

// Get a mutable iterator to the diagonal of an array.
pub trait GetDiagMut {
    type Item;
    type Iter<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    fn diag_iter_mut(&mut self) -> Self::Iter<'_>;
}
