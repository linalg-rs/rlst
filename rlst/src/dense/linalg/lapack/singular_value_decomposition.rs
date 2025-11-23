//! Implementation of the singular value decomposition using LAPACK.

use crate::UnsafeRandom1DAccessByValue;
use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::gesdd::JobZ;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::base::Gemm;
use crate::traits::linalg::decompositions::SingularValueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;

/// Symmetric eigenvalue decomposition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdMode {
    /// Compute full matrices U and V.
    Full,
    /// Compute compact matrices U and V.
    Compact,
}

impl<Item, ArrayImpl> SingularValueDecomposition for Array<ArrayImpl, 2>
where
    Item: Lapack + Gemm,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    fn singular_values(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>> {
        let mut a = DynArray::new_from(self);
        let [m, n] = a.shape();
        let k = std::cmp::min(m, n);

        let mut s = DynArray::<<Self::Item as RlstScalar>::Real, 1>::from_shape([k]);

        Item::gesdd(
            JobZ::N,
            m,
            n,
            a.data_mut().unwrap(),
            m,
            s.data_mut().unwrap(),
            None,
            1,
            None,
            1,
        )?;

        Ok(s)
    }

    fn svd(
        &self,
        mode: SvdMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )> {
        let mut a = DynArray::new_from(self);
        let [m, n] = a.shape();
        let k = std::cmp::min(m, n);
        let mut s = DynArray::<<Self::Item as RlstScalar>::Real, 1>::from_shape([k]);
        let (mut u, mut vt, ldvt) = match mode {
            SvdMode::Full => (
                DynArray::<Self::Item, 2>::from_shape([m, m]),
                DynArray::<Self::Item, 2>::from_shape([n, n]),
                n,
            ),
            SvdMode::Compact => (
                DynArray::<Self::Item, 2>::from_shape([m, k]),
                DynArray::<Self::Item, 2>::from_shape([k, n]),
                k,
            ),
        };

        let jobz = match mode {
            SvdMode::Full => JobZ::A,
            SvdMode::Compact => JobZ::S,
        };

        Item::gesdd(
            jobz,
            m,
            n,
            a.data_mut().unwrap(),
            m,
            s.data_mut().unwrap(),
            u.data_mut(),
            m,
            vt.data_mut(),
            ldvt,
        )?;

        Ok((s, u, vt))
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::base_operations::*;
    use crate::traits::linalg::SymmEig;
    use itertools::izip;

    use paste::paste;

    macro_rules! implement_svd_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {


            #[test]
            fn [<test_singular_values_$scalar>]() {
                let m = 10;
                let n = 5;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let ata = dot!(a.r().conj().transpose().eval(), a.r());

                let s = a.singular_values().unwrap();

                let actual = ata
                    .eigenvaluesh()
                    .unwrap()
                    .unary_op(|v| <<$scalar as RlstScalar>::Real>::sqrt(v))
                    .reverse_axis(0);

                crate::assert_array_relative_eq!(s, actual, $tol);
            }

            #[test]
            fn [<test_svd_thin_compact_$scalar>]() {
                let m = 10;
                let n = 5;
                let k = std::cmp::min(m, n);
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let (s, u, vt) = a.svd(SvdMode::Compact).unwrap();

                let s = {
                    let mut s_mat = DynArray::<$scalar, 2>::from_shape([k, k]);
                    izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                        *v_elem = RlstScalar::from_real(w_elem);
                    });
                    s_mat
                };

                let actual = dot!(u.r(), dot!(s.r(), vt.r()));
                crate::assert_array_relative_eq!(actual, a, $tol);
            }

            #[test]
            fn [<test_svd_thin_full_$scalar>]() {
                let m = 10;
                let n = 5;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let (s, u, vt) = a.svd(SvdMode::Full).unwrap();

                let s = {
                    let mut s_mat = DynArray::<$scalar, 2>::from_shape([m, n]);
                    izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                        *v_elem = RlstScalar::from_real(w_elem);
                    });
                    s_mat
                };

                let actual = dot!(u.r(), dot!(s.r(), vt.r()));
                crate::assert_array_relative_eq!(actual, a, $tol);
            }

            #[test]
            fn [<test_svd_thick_compact_$scalar>]() {
                let m = 5;
                let n = 10;
                let k = std::cmp::min(m, n);
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let (s, u, vt) = a.svd(SvdMode::Compact).unwrap();

                let s = {
                    let mut s_mat = DynArray::<$scalar, 2>::from_shape([k, k]);
                    izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                        *v_elem = RlstScalar::from_real(w_elem);
                    });
                    s_mat
                };

                let actual = dot!(u.r(), dot!(s.r(), vt.r()));
                crate::assert_array_relative_eq!(actual, a, $tol);
            }

            #[test]
            fn [<test_svd_thick_full_$scalar>]() {
                let m = 5;
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let (s, u, vt) = a.svd(SvdMode::Full).unwrap();

                let s = {
                    let mut s_mat = DynArray::<$scalar, 2>::from_shape([m, n]);
                    izip!(s_mat.diag_iter_mut(), s.iter_value()).for_each(|(v_elem, w_elem)| {
                        *v_elem = RlstScalar::from_real(w_elem);
                    });
                    s_mat
                };

                let actual = dot!(u.r(), dot!(s.r(), vt.r()));
                crate::assert_array_relative_eq!(actual, a, $tol);
            }


                    }
        };
    }

    implement_svd_tests!(f32, 1E-4);
    implement_svd_tests!(f64, 1E-10);
    implement_svd_tests!(c32, 1E-4);
    implement_svd_tests!(c64, 1E-10);

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
