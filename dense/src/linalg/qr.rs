//! Interface to QR Decomposition

use super::assert_lapack_stride;
use crate::array::Array;
use itertools::Itertools;
use lapack::{cgeqp3, cunmqr, dgeqp3, dormqr, sgeqp3, sormqr, zgeqp3, zunmqr};

use num::Zero;
use rlst_common::traits::*;
use rlst_common::types::*;

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ApplyQSide {
    Left = b'L',
    Right = b'R',
}

#[derive(Clone, Copy)]
pub enum ApplyQTrans {
    NoTrans,
    ConjTrans,
}

pub struct QRDecomposition<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    tau: Vec<Item>,
    jpvt: Vec<i32>,
}

macro_rules! implement_qr_real {
    ($scalar:ty, $geqp3:expr, $ormqr:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > QRDecomposition<$scalar, ArrayImpl>
        {
            pub fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
                let stride = arr.stride();
                let shape = arr.shape();

                let k = std::cmp::min(shape[0], shape[1]);
                if k == 0 {
                    return Err(RlstError::MatrixIsEmpty((shape[0], shape[1])));
                }

                assert_lapack_stride(stride);

                let m = shape[0] as i32;
                let n = shape[1] as i32;
                let lda = stride[1] as i32;

                let mut jpvt = vec![0 as i32; n as usize];
                let mut tau = vec![<$scalar as Zero>::zero(); k];

                let mut work_query = [<$scalar as Zero>::zero()];
                let lwork = -1;

                let mut info = 0;

                unsafe {
                    $geqp3(
                        m,
                        n,
                        arr.data_mut(),
                        lda,
                        &mut jpvt,
                        &mut tau,
                        &mut work_query,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $geqp3(
                        m,
                        n,
                        arr.data_mut(),
                        lda,
                        &mut jpvt,
                        &mut tau,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(Self { arr, tau, jpvt }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }

            pub fn get_perm(&self) -> Vec<usize> {
                self.jpvt
                    .iter()
                    .map(|&elem| elem as usize - 1)
                    .collect_vec()
            }

            pub fn get_r<
                ArrayImplR: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplR, 2>,
            ) {
                let k = *self.arr.shape().iter().min().unwrap();

                let r_shape = [k, self.arr.shape()[1]];

                assert_eq!(r_shape, arr.shape());

                arr.set_zero();

                for col in 0..r_shape[1] {
                    for row in 0..=std::cmp::min(col, k - 1) {
                        *arr.get_mut([row, col]).unwrap() = self.arr.get_value([row, col]).unwrap();
                    }
                }
            }

            pub fn get_p<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
            ) {
                assert_eq!(arr.shape()[0], arr.shape()[1]);
                assert_eq!(arr.shape()[0], self.arr.shape()[1]);

                for (index, &elem) in self.get_perm().iter().enumerate() {
                    *arr.get_mut([elem, index]).unwrap() = <$scalar as num::One>::one();
                }
            }

            pub fn get_q_alloc<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
            ) -> RlstResult<()> {
                assert_eq!(arr.shape()[0], self.arr.shape()[0]);
                arr.set_identity();

                self.apply_q_alloc(arr, ApplyQSide::Left, ApplyQTrans::NoTrans)
            }

            pub fn apply_q_alloc<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
                side: ApplyQSide,
                trans: ApplyQTrans,
            ) -> RlstResult<()> {
                let m = arr.shape()[0] as i32;
                let n = arr.shape()[1] as i32;

                if std::cmp::min(m, n) == 0 {
                    return Err(RlstError::MatrixIsEmpty((m as usize, n as usize)));
                }

                let trans = match trans {
                    ApplyQTrans::ConjTrans => b'T',
                    ApplyQTrans::NoTrans => b'N',
                };

                let k = self.tau.len() as i32;
                assert!(match side {
                    ApplyQSide::Left => k <= m,
                    ApplyQSide::Right => k <= n,
                });

                let lda = self.arr.stride()[1] as i32;

                assert!(match side {
                    ApplyQSide::Left => lda >= std::cmp::max(1, m),
                    ApplyQSide::Right => lda >= std::cmp::max(1, n),
                });

                assert!(self.arr.shape()[1] as i32 >= k);

                let ldc = arr.stride()[1] as i32;
                assert!(ldc >= std::cmp::max(1, m));

                let mut work_query = [<$scalar as Zero>::zero()];
                let lwork = -1;

                let mut info = 0;

                unsafe {
                    $ormqr(
                        side as u8,
                        trans as u8,
                        m,
                        n,
                        k,
                        self.arr.data(),
                        lda,
                        self.tau.as_slice(),
                        arr.data_mut(),
                        ldc,
                        &mut work_query,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $ormqr(
                        side as u8,
                        trans as u8,
                        m,
                        n,
                        k,
                        self.arr.data(),
                        lda,
                        self.tau.as_slice(),
                        arr.data_mut(),
                        ldc,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(()),
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }

        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>,
            > Array<$scalar, ArrayImpl, 2>
        {
            pub fn into_qr_alloc(self) -> RlstResult<QRDecomposition<$scalar, ArrayImpl>> {
                assert!(!self.is_empty(), "Matrix is empty.");
                QRDecomposition::<$scalar, ArrayImpl>::new(self)
            }
        }
    };
}

macro_rules! implement_qr_complex {
    ($scalar:ty, $geqp3:expr, $ormqr:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > QRDecomposition<$scalar, ArrayImpl>
        {
            pub fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
                let stride = arr.stride();
                let shape = arr.shape();

                let k = std::cmp::min(shape[0], shape[1]);
                if k == 0 {
                    return Err(RlstError::MatrixIsEmpty((shape[0], shape[1])));
                }

                assert_lapack_stride(stride);

                let m = shape[0] as i32;
                let n = shape[1] as i32;
                let lda = stride[1] as i32;

                let mut jpvt = vec![0 as i32; n as usize];
                let mut tau = vec![<$scalar as Zero>::zero(); k];

                let mut rwork = vec![<<$scalar as Scalar>::Real as Zero>::zero(); 2 * n as usize];

                let mut work_query = [<$scalar as Zero>::zero()];
                let lwork = -1;

                let mut info = 0;

                unsafe {
                    $geqp3(
                        m,
                        n,
                        arr.data_mut(),
                        lda,
                        &mut jpvt,
                        &mut tau,
                        &mut work_query,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $geqp3(
                        m,
                        n,
                        arr.data_mut(),
                        lda,
                        &mut jpvt,
                        &mut tau,
                        &mut work,
                        lwork,
                        &mut rwork,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(Self { arr, tau, jpvt }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }

            pub fn get_perm(&self) -> Vec<usize> {
                self.jpvt
                    .iter()
                    .map(|&elem| elem as usize - 1)
                    .collect_vec()
            }

            pub fn get_r<
                ArrayImplR: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplR, 2>,
            ) {
                let k = *self.arr.shape().iter().min().unwrap();

                let r_shape = [k, self.arr.shape()[1]];

                assert_eq!(r_shape, arr.shape());

                arr.set_zero();

                for col in 0..r_shape[1] {
                    for row in 0..=std::cmp::min(col, k - 1) {
                        *arr.get_mut([row, col]).unwrap() = self.arr.get_value([row, col]).unwrap();
                    }
                }
            }

            pub fn get_p<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
            ) {
                assert_eq!(arr.shape()[0], arr.shape()[1]);
                assert_eq!(arr.shape()[0], self.arr.shape()[1]);

                for (index, &elem) in self.get_perm().iter().enumerate() {
                    *arr.get_mut([elem, index]).unwrap() = <$scalar as num::One>::one();
                }
            }

            pub fn get_q_alloc<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
            ) -> RlstResult<()> {
                assert_eq!(arr.shape()[0], self.arr.shape()[0]);
                arr.set_identity();

                self.apply_q_alloc(arr, ApplyQSide::Left, ApplyQTrans::NoTrans)
            }

            pub fn apply_q_alloc<
                ArrayImplQ: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplQ, 2>,
                side: ApplyQSide,
                trans: ApplyQTrans,
            ) -> RlstResult<()> {
                let m = arr.shape()[0] as i32;
                let n = arr.shape()[1] as i32;

                if std::cmp::min(m, n) == 0 {
                    return Err(RlstError::MatrixIsEmpty((m as usize, n as usize)));
                }

                let trans = match trans {
                    ApplyQTrans::ConjTrans => b'C',
                    ApplyQTrans::NoTrans => b'N',
                };

                let k = self.tau.len() as i32;
                assert!(match side {
                    ApplyQSide::Left => k <= m,
                    ApplyQSide::Right => k <= n,
                });

                let lda = self.arr.stride()[1] as i32;

                assert!(match side {
                    ApplyQSide::Left => lda >= std::cmp::max(1, m),
                    ApplyQSide::Right => lda >= std::cmp::max(1, n),
                });

                assert!(self.arr.shape()[1] as i32 >= k);

                let ldc = arr.stride()[1] as i32;
                assert!(ldc >= std::cmp::max(1, m));

                let mut work_query = [<$scalar as Zero>::zero()];
                let lwork = -1;

                let mut info = 0;

                unsafe {
                    $ormqr(
                        side as u8,
                        trans as u8,
                        m,
                        n,
                        k,
                        self.arr.data(),
                        lda,
                        self.tau.as_slice(),
                        arr.data_mut(),
                        ldc,
                        &mut work_query,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;

                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    $ormqr(
                        side as u8,
                        trans as u8,
                        m,
                        n,
                        k,
                        self.arr.data(),
                        lda,
                        self.tau.as_slice(),
                        arr.data_mut(),
                        ldc,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(()),
                    _ => return Err(RlstError::LapackError(info)),
                }
            }
        }

        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>,
            > Array<$scalar, ArrayImpl, 2>
        {
            pub fn into_qr_alloc(self) -> RlstResult<QRDecomposition<$scalar, ArrayImpl>> {
                assert!(!self.is_empty(), "Matrix is empty.");
                QRDecomposition::<$scalar, ArrayImpl>::new(self)
            }
        }
    };
}

implement_qr_real!(f64, dgeqp3, dormqr);
implement_qr_real!(f32, sgeqp3, sormqr);
implement_qr_complex!(c64, zgeqp3, zunmqr);
implement_qr_complex!(c32, cgeqp3, cunmqr);

#[cfg(test)]
mod test {

    use rlst_common::types::*;
    use rlst_common::{assert_array_abs_diff_eq, assert_array_relative_eq, traits::*};

    use crate::array::empty_array;
    use crate::rlst_dynamic_array2;
    use paste::paste;

    macro_rules! implement_qr_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            pub fn [<test_thin_qr_$scalar>]() {
                let shape = [8, 5];
                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                let mut r_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut q_mat = rlst_dynamic_array2!($scalar, [8, 5]);
                let mut p_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut p_trans = rlst_dynamic_array2!($scalar, [5, 5]);
                let actual = rlst_dynamic_array2!($scalar, [8, 5]);
                let mut ident = rlst_dynamic_array2!($scalar, [5, 5]);
                ident.set_identity();

                let qr = mat.into_qr_alloc().unwrap();

                let _ = qr.get_r(r_mat.view_mut());
                let _ = qr.get_q_alloc(q_mat.view_mut());
                let _ = qr.get_p(p_mat.view_mut());

                p_trans.fill_from(p_mat.transpose());

                let actual = empty_array::<$scalar, 2>()
                    .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat.view()), p_trans);

                assert_array_relative_eq!(actual, mat2, $tol);

                let qtq = empty_array::<$scalar, 2>().mult_into_resize(
                    rlst_blis::interface::types::TransMode::ConjTrans,
                    rlst_blis::interface::types::TransMode::NoTrans,
                    1.0.into(),
                    q_mat.view(),
                    q_mat.view(),
                    1.0.into(),
                );

                assert_array_abs_diff_eq!(qtq, ident, $tol);
            }

            #[test]
            pub fn [<test_thick_qr_$scalar>]() {
                let shape = [5, 8];
                let mut mat = rlst_dynamic_array2!($scalar, shape);
                let mut mat2 = rlst_dynamic_array2!($scalar, shape);

                mat.fill_from_seed_equally_distributed(0);
                mat2.fill_from(mat.view());

                let mut r_mat = rlst_dynamic_array2!($scalar, [5, 8]);
                let mut q_mat = rlst_dynamic_array2!($scalar, [5, 5]);
                let mut p_mat = rlst_dynamic_array2!($scalar, [8, 8]);
                let mut p_trans = rlst_dynamic_array2!($scalar, [8, 8]);
                let actual = rlst_dynamic_array2!($scalar, [5, 8]);
                let mut ident = rlst_dynamic_array2!($scalar, [5, 5]);
                ident.set_identity();

                let qr = mat.into_qr_alloc().unwrap();

                let _ = qr.get_r(r_mat.view_mut());
                let _ = qr.get_q_alloc(q_mat.view_mut());
                let _ = qr.get_p(p_mat.view_mut());

                p_trans.fill_from(p_mat.transpose());

                let actual = empty_array::<$scalar, 2>()
                    .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat), p_trans);

                assert_array_relative_eq!(actual, mat2, $tol);

                let qtq = empty_array::<$scalar, 2>().mult_into_resize(
                    rlst_blis::interface::types::TransMode::ConjTrans,
                    rlst_blis::interface::types::TransMode::NoTrans,
                    1.0.into(),
                    q_mat.view(),
                    q_mat.view(),
                    1.0.into(),
                );

                assert_array_abs_diff_eq!(qtq, ident, $tol);
            }

                    }
        };
    }

    implement_qr_tests!(f32, 1E-6);
    implement_qr_tests!(f64, 1E-12);
    implement_qr_tests!(c32, 1E-6);
    implement_qr_tests!(c64, 1E-12);
}
