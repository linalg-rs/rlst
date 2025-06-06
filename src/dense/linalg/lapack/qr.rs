//! Implementation of the QR decomposition.

use crate::dense::array::{Array, DynArray};
use crate::dense::traits::{RawAccessMut, UnsafeRandomAccessMut};
use crate::{dense::types::RlstResult, RawAccess, Shape, Stride};

use crate::dense::types::RlstError;
use crate::dense::types::RlstScalar;

use super::LapackWrapper;

use lapack::dgeqp3;
use num::One;
use num::Zero;

/// Apply Q side
#[derive(Clone, Copy)]
#[repr(u8)]
pub enum ApplyQSide {
    /// Left
    Left = b'L',
    /// Right
    Right = b'R',
}

/// Transpose
#[derive(Clone, Copy)]
pub enum ApplyQTrans {
    /// No transpose
    NoTrans,
    /// Conjugate transpose
    ConjTrans,
}

/// A trait for computing the QR decomposition of a matrix.
pub trait LapackQr {
    /// Output type of the QR decomposition.
    type Output: ComputedQr;

    /// Compute the QR Decomposition of the matrix.
    fn qr(self) -> RlstResult<Self::Output>;
}

/// Operations on a computed QR decomposition.
pub trait ComputedQr {
    /// The item type of the QR decomposition.
    type Item;

    /// Array Implementation type.
    type ArrayImpl: Shape<2> + Stride<2> + RawAccess<Item = Self::Item>;

    /// Return the QR decomposition data.
    fn qr_data(&self) -> &LapackWrapper<Self::Item, Self::ArrayImpl>;
}

/// A struct representing the QR decomposition of a matrix.
pub struct QrDecomposition<Item, ArrayImpl> {
    /// The QR decomposition data.
    qr: LapackWrapper<Item, ArrayImpl>,
    /// The tau vector.
    tau: Vec<Item>,
    /// The pivot indices.
    jpvt: Vec<i32>,
}

macro_rules! implement_lapack_qr_real {
    ($scalar:ty, $geqp3:expr) => {
        impl<ArrayImpl> LapackQr for LapackWrapper<$scalar, ArrayImpl>
        where
            ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = $scalar>,
        {
            type Output = QrDecomposition<$scalar, ArrayImpl>;

            /// Compute the QR decomposition of the matrix.
            fn qr(mut self) -> RlstResult<Self::Output> {
                let (m, n, lda) = self.lapack_dims();
                let k = std::cmp::min(m, n);

                let mut jpvt = vec![0 as i32; n as usize];
                let mut tau = vec![<$scalar as Zero>::zero(); k as usize];

                let mut work_query = [<$scalar as Zero>::zero()];
                let lwork = -1;

                let mut info = 0;

                unsafe {
                    $geqp3(
                        m,
                        n,
                        self.data_mut(),
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
                        self.data_mut(),
                        lda,
                        &mut jpvt,
                        &mut tau,
                        &mut work,
                        lwork,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(QrDecomposition {
                        qr: self,
                        tau,
                        jpvt,
                    }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }
        }
    };
}

implement_lapack_qr_real!(f64, dgeqp3);

macro_rules! implement_qr_ops {
    ($scalar:ty, $ormqr:expr) => {
        impl<ArrayImpl> ComputedQr for QrDecomposition<$scalar, ArrayImpl>
        where
            ArrayImpl: Shape<2> + Stride<2> + RawAccess<Item = $scalar>,
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            /// Return the QR decomposition data.
            fn qr_data(&self) -> &LapackWrapper<Self::Item, Self::ArrayImpl> {
                &self.qr
            }
        }
    };
}

implement_qr_ops!(f64, dormqr);
