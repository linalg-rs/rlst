//! Implementation of the pseudo-inverse.

use itertools::izip;

use crate::{
    dense::array::{Array, DynArray},
    diag, dot,
    traits::{
        array::EvaluateArray,
        iterators::{ArrayIteratorByValue, ArrayIteratorMut, ColumnIteratorMut},
        linalg::{base::Gemm, lapack::Lapack},
        rlst_num::RlstScalar,
    },
};

/// A structure representing the pseudo-inverse of a matrix.
pub struct PInv<Item> {
    s: DynArray<Item, 1>,
    ut: DynArray<Item, 2>,
    v: DynArray<Item, 2>,
}

impl<Item: Lapack + Gemm> PInv<Item> {
    /// Create a new pseudo-inverse from singular values and matrices.
    /// It is assumed that only positive singular values and corresponding
    /// singular vectors are provided.
    pub fn new(s: DynArray<Item, 1>, ut: DynArray<Item, 2>, v: DynArray<Item, 2>) -> Self {
        Self { s, ut, v }
    }

    /// Get the singular values.
    pub fn s(&self) -> &DynArray<Item, 1> {
        &self.s
    }

    /// Get the U^T matrix.
    pub fn ut(&self) -> &DynArray<Item, 2> {
        &self.ut
    }

    /// Get the V matrix.
    pub fn v(&self) -> &DynArray<Item, 2> {
        &self.v
    }

    /// Return the matrix form of the pseudo-inverse
    pub fn as_matrix(&self) -> DynArray<Item, 2> {
        let sinv = self
            .s
            .r()
            .apply_unary_op(|elem| <Item as RlstScalar>::recip(elem));

        dot!(self.v.r(), diag!(sinv), self.ut.r())
    }

    /// Apply the pseudo-inverse to a matrix.
    pub fn apply<ArrayImpl, const NDIM: usize>(
        &self,
        arr: &Array<ArrayImpl, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        Array<ArrayImpl, NDIM>: EvaluateArray<Output = DynArray<Item, NDIM>>,
    {
        let sinv = self
            .s
            .r()
            .apply_unary_op(|elem| <Item as RlstScalar>::recip(elem));

        if NDIM == 2 {
            let arr = arr.eval().coerce_dim::<2>().unwrap();
            let mut tmp = dot!(self.ut.r(), arr);
            for mut col in ColumnIteratorMut::col_iter_mut(&mut tmp) {
                for (elem, si) in izip!(col.iter_mut(), sinv.iter_value()) {
                    *elem *= si;
                }
            }
            dot!(self.v.r(), tmp).coerce_dim::<NDIM>().unwrap().eval()
        } else if NDIM == 1 {
            let arr = arr.eval().coerce_dim::<1>().unwrap();
            let mut tmp = dot!(self.ut.r(), arr);
            for (elem, si) in izip!(tmp.iter_mut(), sinv.iter_value()) {
                *elem *= si;
            }
            dot!(self.v.r(), tmp).coerce_dim::<NDIM>().unwrap().eval()
        } else {
            panic!("`PInv::apply` is only implemented for 1D and 2D arrays.");
        }
    }
}
