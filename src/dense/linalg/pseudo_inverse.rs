//! Pseudo-Inverse of a matrix.

use crate::dense::array::Array;
use crate::dense::traits::{
    MultInto, RawAccessMut, ResizeInPlace, Shape, Stride, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::dense::types::{c32, c64, RlstResult, RlstScalar};
use crate::rlst_dynamic_array2;
use itertools::Itertools;
use num::traits::{One, Zero};

use crate::dense::linalg::svd::MatrixSvd;

/// Pseudo-inverse of a matrix
pub trait MatrixPseudoInverse: RlstScalar + MatrixSvd {
    /// Compute the pseudoinverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. Array is resized if necessary.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol` will be discarded.
    fn into_pseudo_inverse_resize_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>,
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>
            + ResizeInPlace<2>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        pinv: Array<Self, ArrayImplPInv, 2>,
        tol: <Self as RlstScalar>::Real,
    ) -> RlstResult<()>;

    /// Compute the pseudo inverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. If `self` has shape `[m, n]` then
    ///  `pinv` must have shape `[n, m]`.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol * s\[0\]` will be discarded,
    ///  where s\[0\] is the largest singular value.
    fn into_pseudo_inverse_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>,
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        pinv: Array<Self, ArrayImplPInv, 2>,
        tol: <Self as RlstScalar>::Real,
    ) -> RlstResult<()>;
}

impl<
        Item: RlstScalar + MatrixPseudoInverse,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Item>
            + RawAccessMut<Item = Item>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the pseudoinverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. Array is resized if necessary.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol` will be discarded.
    pub fn into_pseudo_inverse_resize_alloc<
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + ResizeInPlace<2>,
    >(
        self,
        pinv: Array<Item, ArrayImplPInv, 2>,
        tol: <Item as RlstScalar>::Real,
    ) -> RlstResult<()> {
        <Item as MatrixPseudoInverse>::into_pseudo_inverse_resize_alloc(self, pinv, tol)
    }

    /// Compute the pseudo inverse into the array `pinv`.
    ///
    /// This method dynamically allocates memory for the computation.
    ///
    /// # Parameters
    /// - `pinv`: Array to store the pseudo-inverse in. If `self` has shape `[m, n]` then
    ///  `pinv` must have shape `[n, m]`.
    /// - `tol`: The relative tolerance. Singular values smaller or equal `tol * s\[0\]` will be discarded,
    ///  where s\[0\] is the largest singular value.
    pub fn into_pseudo_inverse_alloc<
        ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>,
    >(
        self,
        pinv: Array<Item, ArrayImplPInv, 2>,
        tol: <Item as RlstScalar>::Real,
    ) -> RlstResult<()> {
        <Item as MatrixPseudoInverse>::into_pseudo_inverse_alloc(self, pinv, tol)
    }
}

macro_rules! impl_pinv {
    ($scalar:ty) => {
        impl MatrixPseudoInverse for $scalar {
            fn into_pseudo_inverse_resize_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = Self>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as RlstScalar>::Real,
            ) -> RlstResult<()> {
                let shape = pinv.shape();

                if pinv.shape() != [shape[1], shape[0]] {
                    pinv.resize_in_place([shape[1], shape[0]]);
                }
                Self::into_pseudo_inverse_alloc(arr, pinv, tol)
                // arr.into_pseudo_inverse_alloc(pinv, tol)
            }

            fn into_pseudo_inverse_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImpl, 2>,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as RlstScalar>::Real,
            ) -> RlstResult<()> {
                let shape = arr.shape();
                let k = std::cmp::min(arr.shape()[0], arr.shape()[1]);
                let mode = crate::dense::linalg::svd::SvdMode::Reduced;
                let mut singvals = vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); k];
                let mut u = rlst_dynamic_array2!($scalar, [arr.shape()[0], k]);
                let mut vt = rlst_dynamic_array2!($scalar, [k, arr.shape()[1]]);

                arr.into_svd_alloc(u.r_mut(), vt.r_mut(), &mut singvals, mode)?;

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
                    u.r()
                        .transpose()
                        .conj()
                        .into_subview([0, 0], [index, shape[0]]),
                );
                v.fill_from(
                    vt.r()
                        .transpose()
                        .conj()
                        .into_subview([0, 0], [shape[1], index]),
                );

                for (col_index, &singval) in singvals.iter().take(index).enumerate() {
                    v.r_mut().slice(1, col_index).scale_inplace(
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
