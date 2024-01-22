//! Dense matrix traits

pub mod accessors;

pub use crate::linalg::{
    inverse::MatrixInverse, lu::MatrixLuDecomposition, pseudo_inverse::MatrixPseudoInverse,
    qr::MatrixQrDecomposition, svd::MatrixSvd,
};
pub use accessors::*;

use rlst_blis::interface::types::TransMode;
use rlst_common::types::*;

/// Return the shape of the object.
pub trait Shape<const NDIM: usize> {
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

/// Return the stride of the object.
pub trait Stride<const NDIM: usize> {
    fn stride(&self) -> [usize; NDIM];
}

/// Return the number of elements.
pub trait NumberOfElements {
    fn number_of_elements(&self) -> usize;
}

/// Resize an operator in place
pub trait ResizeInPlace<const NDIM: usize> {
    fn resize_in_place(&mut self, shape: [usize; NDIM]);
}

/// Multiply First * Second and sum into Self
pub trait MultInto<First, Second> {
    type Item: RlstScalar;
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

/// Multiply First * Second and sum into Self. Allow to resize Self if necessary
pub trait MultIntoResize<First, Second> {
    type Item: RlstScalar;
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
    type Item: RlstScalar;
    type Iter<'a>: std::iter::Iterator<Item = Self::Item>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;
}

/// Mutable default iterator.
pub trait DefaultIteratorMut {
    type Item: RlstScalar;
    type IterMut<'a>: std::iter::Iterator<Item = &'a mut Self::Item>
    where
        Self: 'a;

    fn iter_mut(&mut self) -> Self::IterMut<'_>;
}

/// Iterate through the elements in `(i, j, data)` form, where
/// `i` is row, `j` is column, and `data` is the corresponding
/// element.
pub trait AijIterator {
    type Item: RlstScalar;
    type Iter<'a>: std::iter::Iterator<Item = (usize, usize, Self::Item)>
    where
        Self: 'a;

    fn iter_aij(&self) -> Self::Iter<'_>;
}
