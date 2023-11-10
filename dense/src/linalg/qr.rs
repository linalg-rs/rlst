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
