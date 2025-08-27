//! Implementation of the pseudo-inverse.

use itertools::izip;

use crate::{
    dense::array::{Array, DynArray},
    diag, dot,
    traits::{
        base_operations::EvaluateObject,
        linalg::{base::Gemm, lapack::Lapack},
        rlst_num::RlstScalar,
    },
    Shape, UnsafeRandom1DAccessByValue,
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
            .unary_op(|elem| <Item as RlstScalar>::recip(elem));

        dot!(self.v.r(), diag!(sinv), self.ut.r())
    }

    /// Apply the pseudo-inverse to a matrix.
    pub fn apply<ArrayImpl, const NDIM: usize>(
        &self,
        arr: &Array<ArrayImpl, NDIM>,
    ) -> DynArray<Item, NDIM>
    where
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    {
        let sinv = self
            .s
            .r()
            .unary_op(|elem| <Item as RlstScalar>::recip(elem));

        if NDIM == 2 {
            let arr = DynArray::new_from(arr).coerce_dim::<2>().unwrap();
            let mut tmp = dot!(self.ut.r(), arr);
            for mut col in tmp.col_iter_mut() {
                for (elem, si) in izip!(col.iter_mut(), sinv.iter_value()) {
                    // Needed because rust analyzer had trouble identifying the type correctly.
                    let elem: &mut Item = elem;
                    *elem *= si;
                }
            }
            dot!(self.v.r(), tmp).coerce_dim::<NDIM>().unwrap().eval()
        } else if NDIM == 1 {
            let arr = DynArray::new_from(arr).coerce_dim::<1>().unwrap();
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::base_operations::*;
    use crate::traits::linalg::SingularvalueDecomposition;
    use itertools::izip;
    use num::Zero;
    use paste::paste;

    macro_rules! implement_pinv_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_pseudo_inverse_thin_$scalar>]() {
                let m = 20;
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_normally_distributed(0);

                let pinv = a.pseudo_inverse(None, None).unwrap();

                let pinv_mat = pinv.as_matrix();

                assert_eq!(pinv_mat.shape(), [n, m]);

                let mut ident = DynArray::<$scalar, 2>::from_shape([n, n]);
                ident.set_identity();

                let actual = dot!(pinv_mat.r(), a.r());

                crate::assert_array_abs_diff_eq!(actual, ident, $tol);

                let mut x = DynArray::<$scalar, 2>::from_shape([m, 2]);

                x.fill_from_seed_equally_distributed(1);

                crate::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
            }

            #[test]
            fn [<test_pseudo_inverse_thick_$scalar>]() {
                let m = 10;
                let n = 20;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_normally_distributed(0);

                let pinv = a.pseudo_inverse(None, None).unwrap();

                let pinv_mat = pinv.as_matrix();

                assert_eq!(pinv_mat.shape(), [n, m]);

                let mut ident = DynArray::<$scalar, 2>::from_shape([m, m]);
                ident.set_identity();

                let actual = dot!(a.r(), pinv_mat.r());

                crate::assert_array_abs_diff_eq!(actual, ident, $tol);

                let mut x = DynArray::<$scalar, 2>::from_shape([m, 2]);

                x.fill_from_seed_equally_distributed(1);

                crate::assert_array_relative_eq!(dot!(pinv_mat.r(), x.r()), pinv.apply(&x), $tol);
            }

                    }
        };
    }

    implement_pinv_tests!(f32, 1E-4);
    implement_pinv_tests!(f64, 1E-10);
    implement_pinv_tests!(c32, 1E-4);
    implement_pinv_tests!(c64, 1E-10);
}
