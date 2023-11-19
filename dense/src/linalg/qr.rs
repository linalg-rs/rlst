//! Interface to QR Decomposition

use super::assert_lapack_stride;
use crate::array::Array;
use itertools::Itertools;
use lapack::{dgeqp3, dormqr};

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
#[repr(u8)]
pub enum ApplyQTrans {
    NoTrans = b'N',
    Trans = b'T',
}

pub struct QRDecomposition<
    Item: Scalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    tau: Vec<Item>,
    jpvt: Vec<i32>,
}

impl<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = f64> + Stride<2> + Shape<2> + RawAccessMut<Item = f64>,
    > QRDecomposition<f64, ArrayImpl>
{
    pub fn new(mut arr: Array<f64, ArrayImpl, 2>) -> RlstResult<Self> {
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
        let mut tau = vec![<f64 as Zero>::zero(); k];

        let mut work_query = [<f64 as Zero>::zero()];
        let lwork = -1;

        let mut info = 0;

        unsafe {
            dgeqp3(
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
        let mut work = vec![<f64 as Zero>::zero(); lwork as usize];

        unsafe {
            dgeqp3(
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
        ArrayImplR: UnsafeRandomAccessByValue<2, Item = f64>
            + UnsafeRandomAccessMut<2, Item = f64>
            + RawAccessMut<Item = f64>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        mut arr: Array<f64, ArrayImplR, 2>,
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
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = f64>
            + UnsafeRandomAccessMut<2, Item = f64>
            + RawAccessMut<Item = f64>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        mut arr: Array<f64, ArrayImplQ, 2>,
    ) {
        assert_eq!(arr.shape()[0], arr.shape()[1]);
        assert_eq!(arr.shape()[0], self.arr.shape()[1]);

        for (index, &elem) in self.get_perm().iter().enumerate() {
            *arr.get_mut([elem, index]).unwrap() = <f64 as num::One>::one();
        }
    }

    pub fn get_q<
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = f64>
            + UnsafeRandomAccessMut<2, Item = f64>
            + RawAccessMut<Item = f64>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        mut arr: Array<f64, ArrayImplQ, 2>,
    ) -> RlstResult<()> {
        assert_eq!(arr.shape()[0], self.arr.shape()[0]);
        arr.set_identity();

        self.apply_q(arr, ApplyQSide::Left, ApplyQTrans::NoTrans)
    }

    pub fn apply_q<
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = f64>
            + UnsafeRandomAccessMut<2, Item = f64>
            + RawAccessMut<Item = f64>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        mut arr: Array<f64, ArrayImplQ, 2>,
        side: ApplyQSide,
        trans: ApplyQTrans,
    ) -> RlstResult<()> {
        let m = arr.shape()[0] as i32;
        let n = arr.shape()[1] as i32;

        if std::cmp::min(m, n) == 0 {
            return Err(RlstError::MatrixIsEmpty((m as usize, n as usize)));
        }

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

        let mut work_query = [<f64 as Zero>::zero()];
        let lwork = -1;

        let mut info = 0;

        unsafe {
            dormqr(
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

        let mut work = vec![<f64 as Zero>::zero(); lwork as usize];

        unsafe {
            dormqr(
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
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = f64> + Stride<2> + RawAccessMut<Item = f64> + Shape<2>,
    > Array<f64, ArrayImpl, 2>
{
    pub fn into_qr(self) -> RlstResult<QRDecomposition<f64, ArrayImpl>> {
        QRDecomposition::new(self)
    }
}

#[cfg(test)]
mod test {

    use rlst_common::{assert_array_abs_diff_eq, assert_array_relative_eq, traits::*};

    use crate::array::empty_array;
    use crate::rlst_dynamic_array2;

    #[test]
    pub fn test_thin_qr() {
        let shape = [8, 5];
        let mut mat = rlst_dynamic_array2!(f64, shape);
        let mut mat2 = rlst_dynamic_array2!(f64, shape);

        mat.fill_from_seed_equally_distributed(0);
        mat2.fill_from(mat.view());

        let mut r_mat = rlst_dynamic_array2!(f64, [5, 5]);
        let mut q_mat = rlst_dynamic_array2!(f64, [8, 5]);
        let mut p_mat = rlst_dynamic_array2!(f64, [5, 5]);
        let mut p_trans = rlst_dynamic_array2!(f64, [5, 5]);
        let actual = rlst_dynamic_array2!(f64, [8, 5]);
        let mut ident = rlst_dynamic_array2!(f64, [5, 5]);
        ident.set_identity();

        let qr = mat.into_qr().unwrap();

        let _ = qr.get_r(r_mat.view_mut());
        let _ = qr.get_q(q_mat.view_mut());
        let _ = qr.get_p(p_mat.view_mut());

        p_trans.fill_from(p_mat.transpose());

        let actual = empty_array::<f64, 2>()
            .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat.view()), p_trans);

        assert_array_relative_eq!(actual, mat2, 1E-13);

        let qtq = empty_array::<f64, 2>().mult_into_resize(
            rlst_blis::interface::types::TransMode::Trans,
            rlst_blis::interface::types::TransMode::NoTrans,
            1.0,
            q_mat.view(),
            q_mat.view(),
            1.0,
        );

        assert_array_abs_diff_eq!(qtq, ident, 1E-13);
    }

    #[test]
    pub fn test_thick_qr() {
        let shape = [5, 8];
        let mut mat = rlst_dynamic_array2!(f64, shape);
        let mut mat2 = rlst_dynamic_array2!(f64, shape);

        mat.fill_from_seed_equally_distributed(0);
        mat2.fill_from(mat.view());

        let mut r_mat = rlst_dynamic_array2!(f64, [5, 8]);
        let mut q_mat = rlst_dynamic_array2!(f64, [5, 5]);
        let mut p_mat = rlst_dynamic_array2!(f64, [8, 8]);
        let mut p_trans = rlst_dynamic_array2!(f64, [8, 8]);
        let actual = rlst_dynamic_array2!(f64, [5, 8]);
        let mut ident = rlst_dynamic_array2!(f64, [5, 5]);
        ident.set_identity();

        let qr = mat.into_qr().unwrap();

        let _ = qr.get_r(r_mat.view_mut());
        let _ = qr.get_q(q_mat.view_mut());
        let _ = qr.get_p(p_mat.view_mut());

        p_trans.fill_from(p_mat.transpose());

        let actual = empty_array::<f64, 2>()
            .simple_mult_into_resize(actual.simple_mult_into(q_mat.view(), r_mat), p_trans);

        assert_array_relative_eq!(actual, mat2, 1E-13);

        let qtq = empty_array::<f64, 2>().mult_into_resize(
            rlst_blis::interface::types::TransMode::Trans,
            rlst_blis::interface::types::TransMode::NoTrans,
            1.0,
            q_mat.view(),
            q_mat.view(),
            1.0,
        );

        assert_array_abs_diff_eq!(qtq, ident, 1E-13);
    }
}
