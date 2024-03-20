//! Pseudo-Inverse of a matrix.

use crate::array::Array;
use crate::rlst_dynamic_array2;
use crate::traits::*;
use crate::types::{c32, c64, RlstResult, RlstScalar};
use itertools::Itertools;
use num::traits::{One, Zero};

use crate::linalg::svd::*;

/// Pseudo-inverse of a matrix
pub trait MatrixPseudoInverse {
    /// Item type
    type Item: RlstScalar;

    /// Compute the pseudo inverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. Array is resized if necessary.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol` will be discarded.
    fn into_pseudo_inverse_resize_alloc<
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + ResizeInPlace<2>,
    >(
        self,
        pinv: Array<Self::Item, ArrayImplPInv, 2>,
        tol: <Self::Item as RlstScalar>::Real,
    ) -> RlstResult<()>;

    /// Compute the pseudo inverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. If `self` has shape `[m, n]` then
    ///           `pinv` must have shape `[n, m]`.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol * s\[0\]` will be discarded,
    /// where s\[0\] is the largest singular value.
    fn into_pseudo_inverse_alloc<
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>,
    >(
        self,
        pinv: Array<Self::Item, ArrayImplPInv, 2>,
        tol: <Self::Item as RlstScalar>::Real,
    ) -> RlstResult<()>;
}

macro_rules! impl_pinv {
    ($scalar:ty) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixPseudoInverse for Array<$scalar, ArrayImpl, 2>
        {
            type Item = $scalar;

            fn into_pseudo_inverse_resize_alloc<
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                self,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as RlstScalar>::Real,
            ) -> RlstResult<()> {
                let shape = pinv.shape();

                if pinv.shape() != [shape[1], shape[0]] {
                    pinv.resize_in_place([shape[1], shape[0]]);
                }
                self.into_pseudo_inverse_alloc(pinv, tol)
            }

            fn into_pseudo_inverse_alloc<
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>,
            >(
                self,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as RlstScalar>::Real,
            ) -> RlstResult<()> {
                let shape = self.shape();
                let k = std::cmp::min(self.shape()[0], self.shape()[1]);
                let mode = crate::linalg::svd::SvdMode::Reduced;
                let mut singvals = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); k];
                let mut u = rlst_dynamic_array2!($scalar, [self.shape()[0], k]);
                let mut vt = rlst_dynamic_array2!($scalar, [k, self.shape()[1]]);

                self.into_svd_alloc(u.view_mut(), vt.view_mut(), &mut singvals, mode)?;

                let index = match singvals
                    .iter()
                    .find_position(|&&elem| elem <= tol * singvals[0])
                {
                    Some((index, _)) => index,
                    None => k,
                };

                if index == 0 {
                    pinv.set_zero();
                    return Ok(());
                }

                let mut uh = rlst_dynamic_array2!($scalar, [index, shape[0]]);
                let mut v = rlst_dynamic_array2!($scalar, [shape[1], index]);

                uh.fill_from(
                    u.view()
                        .transpose()
                        .conj()
                        .into_subview([0, 0], [index, shape[0]]),
                );
                v.fill_from(
                    vt.view()
                        .transpose()
                        .conj()
                        .into_subview([0, 0], [shape[1], index]),
                );

                for (col_index, &singval) in singvals.iter().take(index).enumerate() {
                    v.view_mut().slice(1, col_index).scale_in_place(
                        (<<$scalar as RlstScalar>::Real as One>::one() / singval).into(),
                    );
                }

                pinv.simple_mult_into(v, uh);

                Ok(())
            }
        }
    };
}

impl_pinv!(f64);
impl_pinv!(f32);
impl_pinv!(c32);
impl_pinv!(c64);
