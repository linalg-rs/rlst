//! Dense matrix traits

pub mod accessors;

pub use accessors::*;
use typenum::Unsigned;

use crate::dense::types::TransMode;

use super::types::RlstNum;

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

/// Multiply into with resize
pub trait MultIntoResize<First, Second> {
    /// Item type
    type Item: RlstNum;
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
    type Item;
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
    type Item;
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
    type Item;
    /// Iterator
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::Item)>
    where
        Self: 'a;
    /// Get iterator
    fn iter_aij(&self) -> Self::Iter<'_>;
}

// /// Helper trait that returns from an enumeration iterator a new iterator
// /// that converts the 1d index into a multi-index.
// pub trait AsMultiIndex<T, I: Iterator<Item = (usize, T)>, const NDIM: usize> {
//     /// Get multi-index
//     fn multi_index(
//         self,
//         shape: [usize; NDIM],
//     ) -> crate::dense::array::iterators::MultiIndexIterator<T, I, NDIM>;
// }

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
pub trait ArrayIteratorMut: ArrayIterator {
    /// Type of the iterator.
    type IterMut<'a>: Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    /// Returns an iterator that produces mutable references to elements of type `Self::Item`.
    fn iter_mut(&self) -> Self::IterMut<'_>;
}

// /// Basic trait for arrays that provide random access to values.
// pub trait ValueArrayImpl<const NDIM: usize, Item: RlstBase>:
//     UnsafeRandomAccessByValue<NDIM, Item = Item>
//     + Shape<NDIM>
//     + UnsafeRandom1DAccessByValue<Item = Item>
// {
// }

// /// Basic trait for arrays that provide mutable random access to values.
// pub trait MutableArrayImpl<const NDIM: usize, Item: RlstBase>:
//     ValueArrayImpl<NDIM, Item>
//     + UnsafeRandomAccessMut<NDIM, Item = Item>
//     + UnsafeRandom1DAccessMut<Item = Item>
// {
// }

// /// Basic trait for arrays that provide access by reference
// pub trait RefArrayImpl<const NDIM: usize, Item: RlstBase>:
//     ValueArrayImpl<NDIM, Item>
//     + UnsafeRandomAccessByRef<NDIM, Item = Item>
//     + UnsafeRandom1DAccessByRef<Item = Item>
// {
// }

// impl<const NDIM: usize, Item: RlstBase, ArrayImpl> ValueArrayImpl<NDIM, Item> for ArrayImpl where
//     ArrayImpl: UnsafeRandomAccessByValue<NDIM, Item = Item>
//         + UnsafeRandom1DAccessByValue<Item = Item>
//         + Shape<NDIM>
// {
// }

// impl<const NDIM: usize, Item: RlstBase, ArrayImpl> RefArrayImpl<NDIM, Item> for ArrayImpl where
//     ArrayImpl: ValueArrayImpl<NDIM, Item>
//         + UnsafeRandomAccessByRef<NDIM, Item = Item>
//         + UnsafeRandom1DAccessByRef<Item = Item>
// {
// }

// impl<const NDIM: usize, Item: RlstBase, ArrayImpl> MutableArrayImpl<NDIM, Item> for ArrayImpl where
//     ArrayImpl: ValueArrayImpl<NDIM, Item>
//         + UnsafeRandomAccessMut<NDIM, Item = Item>
//         + UnsafeRandom1DAccessMut<Item = Item>
// {
// }
