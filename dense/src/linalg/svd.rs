//! Implement the SVD
use crate::array::Array;
use lapack::{cgesvd, dgesvd, sgesvd, zgesvd};
use num::traits::Zero;
use rlst_common::traits::*;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};

use super::assert_lapack_stride;

pub enum SvdMode {
    Reduced,
    Full,
}

macro_rules! impl_svd_real {
    ($scalar:ty, $gesvd:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > Array<$scalar, ArrayImpl, 2>
        {
            pub fn into_singular_values(
                mut self,
                singular_values: &mut [<$scalar as Scalar>::Real],
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                assert_lapack_stride(self.stride());
                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);
                let lda = self.stride()[1] as i32;
                let mut u = [<$scalar as Zero>::zero(); 1];
                let mut vt = [<$scalar as Zero>::zero(); 1];
                let ldu = 1;
                let ldvt = 1;
                let mut info = 0;

                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        &mut u,
                        ldu,
                        &mut vt,
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        &mut u,
                        ldu,
                        &mut vt,
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }

            pub fn into_svd<
                ArrayImplU: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplVt: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut self,
                mut u: Array<$scalar, ArrayImplU, 2>,
                mut vt: Array<$scalar, ArrayImplVt, 2>,
                singular_values: &mut [<$scalar as Scalar>::Real],
                mode: SvdMode,
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                let jobu;
                let jobvt;

                assert_lapack_stride(self.stride());
                assert_lapack_stride(u.stride());
                assert_lapack_stride(u.stride());

                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);

                match mode {
                    SvdMode::Full => {
                        jobu = b'A';
                        jobvt = b'A';

                        assert_eq!(u.shape(), [m as usize, m as usize]);
                        assert_eq!(vt.shape(), [n as usize, n as usize]);
                    }
                    SvdMode::Reduced => {
                        jobu = b'S';
                        jobvt = b'S';

                        assert_eq!(u.shape(), [m as usize, k as usize]);
                        assert_eq!(vt.shape(), [k as usize, n as usize]);
                    }
                }

                let lda = self.stride()[1] as i32;
                let ldu = u.stride()[1] as i32;
                let ldvt = vt.stride()[1] as i32;
                let mut info = 0;

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

macro_rules! impl_svd_complex {
    ($scalar:ty, $gesvd:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > Array<$scalar, ArrayImpl, 2>
        {
            pub fn into_singular_values(
                mut self,
                singular_values: &mut [<$scalar as Scalar>::Real],
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];

                assert_lapack_stride(self.stride());
                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);
                let lda = self.stride()[1] as i32;
                let mut u = [<$scalar as Zero>::zero(); 1];
                let mut vt = [<$scalar as Zero>::zero(); 1];
                let ldu = 1;
                let ldvt = 1;
                let mut info = 0;

                let mut rwork = vec![<<$scalar as Scalar>::Real as Zero>::zero(); 5 * k as usize];

                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        &mut u,
                        ldu,
                        &mut vt,
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        b'N',
                        b'N',
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        &mut u,
                        ldu,
                        &mut vt,
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }

            pub fn into_svd<
                ArrayImplU: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
                ArrayImplVt: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut self,
                mut u: Array<$scalar, ArrayImplU, 2>,
                mut vt: Array<$scalar, ArrayImplVt, 2>,
                singular_values: &mut [<$scalar as Scalar>::Real],
                mode: SvdMode,
            ) -> RlstResult<()> {
                let lwork: i32 = -1;
                let mut work = [<$scalar as Zero>::zero(); 1];
                let jobu;
                let jobvt;

                assert_lapack_stride(self.stride());
                assert_lapack_stride(u.stride());
                assert_lapack_stride(u.stride());

                let m = self.shape()[0] as i32;
                let n = self.shape()[1] as i32;
                let k = std::cmp::min(m, n);
                assert_eq!(k, singular_values.len() as i32);

                match mode {
                    SvdMode::Full => {
                        jobu = b'A';
                        jobvt = b'A';

                        assert_eq!(u.shape(), [m as usize, m as usize]);
                        assert_eq!(vt.shape(), [n as usize, n as usize]);
                    }
                    SvdMode::Reduced => {
                        jobu = b'S';
                        jobvt = b'S';

                        assert_eq!(u.shape(), [m as usize, k as usize]);
                        assert_eq!(vt.shape(), [k as usize, n as usize]);
                    }
                }

                let lda = self.stride()[1] as i32;
                let ldu = u.stride()[1] as i32;
                let ldvt = vt.stride()[1] as i32;
                let mut info = 0;

                let mut rwork = vec![<<$scalar as Scalar>::Real as Zero>::zero(); 5 * k as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    return Err(RlstError::LapackError(info));
                }

                let lwork = work[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $gesvd(
                        jobu,
                        jobvt,
                        m,
                        n,
                        self.data_mut(),
                        lda,
                        singular_values,
                        u.data_mut(),
                        ldu,
                        vt.data_mut(),
                        ldvt,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                if info != 0 {
                    Err(RlstError::LapackError(info))
                } else {
                    Ok(())
                }
            }
        }
    };
}

impl_svd_real!(f64, dgesvd);
impl_svd_real!(f32, sgesvd);
impl_svd_complex!(c32, cgesvd);
impl_svd_complex!(c64, zgesvd);

#[cfg(test)]
mod test {

    use super::*;

    use approx::assert_relative_eq;
    use paste::paste;
    use rlst_common::assert_array_relative_eq;

    use crate::array::empty_array;
    use crate::{rlst_dynamic_array1, rlst_dynamic_array2};

    macro_rules! impl_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

                #[test]
                fn [<test_singular_values_$scalar>]() {
                    [<test_singular_values_impl_$scalar>](5, 10, $tol);
                    [<test_singular_values_impl_$scalar>](10, 5, $tol);
                }

                #[test]
                fn [<test_svd_$scalar>]() {
                    [<test_svd_impl_$scalar>](10, 5, SvdMode::Reduced, $tol);
                    [<test_svd_impl_$scalar>](5, 10, SvdMode::Reduced, $tol);
                    [<test_svd_impl_$scalar>](10, 5, SvdMode::Full, $tol);
                    [<test_svd_impl_$scalar>](5, 10, SvdMode::Full, $tol);
                }

                fn [<test_singular_values_impl_$scalar>](m: usize, n: usize, tol: <$scalar as Scalar>::Real) {
                    let k = std::cmp::min(m, n);
                    let mut mat = rlst_dynamic_array2!($scalar, [m, m]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, m]);
                    let mut sigma = rlst_dynamic_array2!($scalar, [m, n]);

                    mat.fill_from_seed_equally_distributed(0);
                    let qr = mat.into_qr().unwrap();
                    qr.get_q(q.view_mut()).unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = ((k - index) as <$scalar as Scalar>::Real).into();
                    }

                    let a = empty_array::<$scalar, 2>().simple_mult_into_resize(q.view(), sigma.view());

                    let mut singvals = rlst_dynamic_array1!(<$scalar as Scalar>::Real, [k]);

                    a.into_singular_values(singvals.data_mut()).unwrap();

                    for index in 0..k {
                        assert_relative_eq!(singvals[[index]], sigma[[index, index]].re(), epsilon = tol);
                    }
                }

                fn [<test_svd_impl_$scalar>](m: usize, n: usize, mode: SvdMode, tol: <$scalar as Scalar>::Real) {
                    let k = std::cmp::min(m, n);

                    let mut mat_u;
                    let mut u;
                    let mut mat_vt;
                    let mut vt;
                    let mut sigma;

                    match mode {
                        SvdMode::Full => {
                            mat_u = rlst_dynamic_array2!($scalar, [m, m]);
                            u = rlst_dynamic_array2!($scalar, [m, m]);
                            mat_vt = rlst_dynamic_array2!($scalar, [n, n]);
                            vt = rlst_dynamic_array2!($scalar, [n, n]);
                            sigma = rlst_dynamic_array2!($scalar, [m, n]);
                        }
                        SvdMode::Reduced => {
                            mat_u = rlst_dynamic_array2!($scalar, [m, k]);
                            u = rlst_dynamic_array2!($scalar, [m, k]);
                            mat_vt = rlst_dynamic_array2!($scalar, [k, n]);
                            vt = rlst_dynamic_array2!($scalar, [k, n]);
                            sigma = rlst_dynamic_array2!($scalar, [k, k]);
                        }
                    }

                    mat_u.fill_from_seed_equally_distributed(0);
                    mat_vt.fill_from_seed_equally_distributed(1);

                    let qr = mat_u.into_qr().unwrap();
                    qr.get_q(u.view_mut()).unwrap();

                    let qr = mat_vt.into_qr().unwrap();
                    qr.get_q(vt.view_mut()).unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = ((k - index) as <$scalar as Scalar>::Real).into();
                    }

                    let a = empty_array::<$scalar, 2>().simple_mult_into_resize(
                        empty_array::<$scalar, 2>().simple_mult_into_resize(u.view(), sigma.view()),
                        vt.view(),
                    );

                    let mut expected = rlst_dynamic_array2!($scalar, a.shape());
                    expected.fill_from(a.view());

                    u.set_zero();
                    vt.set_zero();
                    sigma.set_zero();

                    let mut singvals = rlst_dynamic_array1!(<$scalar as Scalar>::Real, [k]);

                    a.into_svd(u.view_mut(), vt.view_mut(), singvals.data_mut(), mode)
                        .unwrap();

                    for index in 0..k {
                        sigma[[index, index]] = singvals[[index]].into();
                    }

                    let actual = empty_array::<$scalar, 2>().simple_mult_into_resize(
                        empty_array::<$scalar, 2>().simple_mult_into_resize(u, sigma),
                        vt,
                    );

                    assert_array_relative_eq!(expected, actual, tol);
                }


            }
        };
    }

    impl_tests!(f32, 1E-5);
    impl_tests!(f64, 1E-12);
    impl_tests!(c32, 1E-6);
    impl_tests!(c64, 1E-12);
}
