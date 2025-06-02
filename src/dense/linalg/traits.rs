//! Traits for linear algebra operations on matrices.

use crate::dense::{
    array::Array,
    traits::{RawAccessMut, Shape, Stride},
    types::RlstResult,
};

// /// Compute the matrix inverse.
// ///
// /// The matrix inverse is defined for a two dimensional square array `arr` of
// /// shape `[m, m]`.
// ///
// /// # Example
// ///
// /// The following command computes the inverse of an array `a`. The content
// /// of `a` is replaced by the inverse.
// /// ```
// /// # use rlst::rlst_dynamic_array;
// /// # use rlst::dense::linalg::inverse::MatrixInverse;
// /// # let mut a = rlst_dynamic_array2!(f64, [3, 3]);
// /// # a.fill_from_seed_equally_distributed(0);
// /// a.r_mut().into_inverse_alloc().unwrap();
// /// ```
// /// This method allocates memory for the inverse computation.
// pub trait MatrixInverse {
//     /// Array implementation type.
//     type ArrayImpl;
//     /// Compute the matrix inverse
//     fn into_inverse(arr: Array<Self::ArrayImpl, 2>) -> RlstResult<()>;
// }
