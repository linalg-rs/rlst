//! Implement the Pseudo-Inverse.

use crate::array::Array;
use crate::rlst_dynamic_array2;
use crate::traits::*;
use itertools::Itertools;
use num::traits::{One, Zero};
use rlst_common::types::{c32, c64, RlstResult, Scalar};

macro_rules! impl_pinv {
    ($scalar:ty) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > Array<$scalar, ArrayImpl, 2>
        {
            /// Compute the pseudo inverse into the array `pinv`.
            ///
            /// This method dynamically allocates memory for the computation.
            ///
            /// # Parameters
            /// - `pinv`: Array to store the pseudo-inverse in. Array is resized if necessary.
            /// - `tol`: The relative tolerance. Singular values smaller or equal `tol` will be discarded.
            pub fn into_pseudo_inverse_resize_alloc<
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                self,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as Scalar>::Real,
            ) -> RlstResult<()> {
                let shape = pinv.shape();

                if pinv.shape() != [shape[1], shape[0]] {
                    pinv.resize_in_place([shape[1], shape[0]]);
                }
                self.into_pseudo_inverse_alloc(pinv, tol)
            }

            /// Compute the pseudo inverse into the array `pinv`.
            ///
            /// This method dynamically allocates memory for the computation.
            ///
            /// # Parameters
            /// - `pinv`: Array to store the pseudo-inverse in. If `self` has shape `[m, n]` then
            ///           `pinv` must have shape `[n, m]`.
            /// - `tol`: The relative tolerance. Singular values smaller or equal `tol` will be discarded.
            pub fn into_pseudo_inverse_alloc<
                ArrayImplPInv: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>,
            >(
                self,
                mut pinv: Array<$scalar, ArrayImplPInv, 2>,
                tol: <$scalar as Scalar>::Real,
            ) -> RlstResult<()> {
                let shape = self.shape();
                let k = std::cmp::min(self.shape()[0], self.shape()[1]);
                let mode = crate::linalg::svd::SvdMode::Reduced;
                let mut singvals = vec![<<$scalar as Scalar>::Real as Zero>::zero(); k];
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
                        (<<$scalar as Scalar>::Real as One>::one() / singval).into(),
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

#[cfg(test)]
mod test {
    use super::*;

    use crate::array::empty_array;
    use crate::assert_array_abs_diff_eq;
    use paste::paste;

    use crate::rlst_dynamic_array2;

    macro_rules! impl_pinv_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

                #[test]
                fn [<test_thick_pinv_$scalar>]() {
                    let shape = [5, 10];
                    let tol = 0.0;

                    let mut mat = rlst_dynamic_array2!($scalar, shape);
                    let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                    let mut pinv = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                    let mut ident = rlst_dynamic_array2!($scalar, [shape[0], shape[0]]);
                    ident.set_identity();

                    mat.fill_from_seed_equally_distributed(0);
                    mat2.fill_from(mat.view());

                    mat2.into_pseudo_inverse_alloc(pinv.view_mut(), tol)
                        .unwrap();

                    let actual = if shape[0] >= shape[1] {
                        empty_array::<$scalar, 2>().simple_mult_into_resize(pinv, mat)
                    } else {
                        empty_array::<$scalar, 2>().simple_mult_into_resize(mat, pinv)
                    };

                    assert_array_abs_diff_eq!(actual, ident, $tol);
                }

                #[test]
                fn [<test_thin_pinv_$scalar>]() {
                    let shape = [10, 5];
                    let tol = 0.0;

                    let mut mat = rlst_dynamic_array2!($scalar, shape);
                    let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                    let mut pinv = rlst_dynamic_array2!($scalar, [shape[1], shape[0]]);
                    let mut ident = rlst_dynamic_array2!($scalar, [shape[1], shape[1]]);
                    ident.set_identity();

                    mat.fill_from_seed_equally_distributed(0);
                    mat2.fill_from(mat.view());

                    mat2.into_pseudo_inverse_alloc(pinv.view_mut(), tol)
                        .unwrap();

                    let actual = if shape[0] >= shape[1] {
                        empty_array::<$scalar, 2>().simple_mult_into_resize(pinv, mat)
                    } else {
                        empty_array::<$scalar, 2>().simple_mult_into_resize(mat, pinv)
                    };

                    assert_array_abs_diff_eq!(actual, ident, $tol);
                }
            }
        };
    }

    impl_pinv_tests!(f32, 1E-5);
    impl_pinv_tests!(f64, 1E-12);
    impl_pinv_tests!(c32, 1E-5);
    impl_pinv_tests!(c64, 1E-12);
}
