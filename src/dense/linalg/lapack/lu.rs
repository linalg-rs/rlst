//! Lapack LU Decomposition.

use super::LapackArrayMut;

/// Container for the LU Decomposition of a matrix.
pub struct LuDecomposition<'a, Item> {
    arr: LapackArrayMut<'a, Item>,
    ipiv: Vec<i32>,
}

macro_rules! implement_lu {
    ($scalar:ty) => {};
}
