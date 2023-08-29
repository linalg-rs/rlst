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

use paste::paste;

use crate::{GenericBaseMatrix, Matrix};

/// Fixed Dimension 2.
pub struct Fixed2;

/// Fixed Dimension 3.
pub struct Fixed3;

/// Dimension determined at runtime.
pub struct Dynamic;

/// This trait provides a constant [SizeIdentifier::N] from
/// which returns a dimension parameter. For `N` > 0 it specifies
/// a compile time dimension. In the case `N` == 0 the dimension is
/// a runtime parameter and not known at compile time.
pub trait SizeIdentifier {
    const SIZE: SizeIdentifierValue;
}

pub trait StaticMatrixBuilder: SizeIdentifier + Sized {
    const N: usize;
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

pub enum SizeIdentifierValue {
    Dynamic,
    Static(usize, usize),
}
