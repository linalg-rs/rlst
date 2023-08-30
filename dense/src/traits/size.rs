//! Definition of Size Traits.
//!
//! Each matrix in RLST has two size identifiers, one for the rows
//! and one for the columns. The size identifiers determine if a row/column
//! size is fixed at compile time or dynamically determined at runtime.
//! The [SizeType] trait implements this functionality. It has two associated types,
//! the row size [SizeType::R] and the column size [SizeType::C]. Both are expected to
//! be types that implement the [SizeIdentifier] trait. A [SizeIdentifier] has an associated
//! [usize] constant [SizeIdentifier::N] that gives compile time information on the actual
//! size. The following data types are implemented that have size types
//!
//! - [Fixed1]. This type specifies a row/column of fixed dimension 1.
//! - [Fixed2]. This type specifies a row/column of fixed dimension 2.
//! - [Fixed3]. This type specifies a row/column of fixed dimension 3.
//! - [Dynamic]. This type specifies a row/column dimension defined at runtime.
//!             The corresponding constant [SizeIdentifier::N] is set to 0.
//!

use crate::{base_matrix::BaseMatrix, DefaultLayout, LayoutType, VectorContainer};
use rlst_common::types::Scalar;

/// Fixed Dimension 2.
pub struct Fixed2;

/// Fixed Dimension 3.
pub struct Fixed3;

/// Dimension determined at runtime.
pub struct Dynamic;

pub trait SizeIdentifier {
    const SIZE: SizeIdentifierValue;
}

pub trait MatrixBuilder<T: Scalar> {
    type Out;

    fn new_matrix(dim: (usize, usize)) -> Self::Out;
}

pub trait Size {
    type S: SizeIdentifier;
}

impl SizeIdentifier for Fixed2 {
    const SIZE: SizeIdentifierValue = SizeIdentifierValue::Static(2, 2);
}

impl SizeIdentifier for Fixed3 {
    const SIZE: SizeIdentifierValue = SizeIdentifierValue::Static(3, 3);
}

impl SizeIdentifier for Dynamic {
    const SIZE: SizeIdentifierValue = SizeIdentifierValue::Dynamic;
}

impl<T: Scalar> MatrixBuilder<T> for Dynamic {
    type Out = crate::MatrixD<T>;

    fn new_matrix(dim: (usize, usize)) -> Self::Out {
        <crate::MatrixD<T>>::new(BaseMatrix::new(
            VectorContainer::new(dim.0 * dim.1),
            DefaultLayout::from_dimension(dim),
        ))
    }
}

pub enum SizeIdentifierValue {
    Dynamic,
    Static(usize, usize),
}
