//! Pivoted QR Decomposition.

use super::assert_lapack_stride;
use crate::dense::array::Array;
use crate::dense::traits::{
    RandomAccessByValue, RandomAccessMut, RawAccess, RawAccessMut, Shape, Stride,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use itertools::Itertools;
use lapack::{cgeqp3, cunmqr, dgeqp3, dormqr, sgeqp3, sormqr, zgeqp3, zunmqr};

use crate::dense::types::{c32, c64, RlstError, RlstResult, RlstScalar};
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

/// Compute the QR decomposition of a given `(m, n)` matrix A.
///
/// The function computes a decomposition of the form `AP=QR`, where
/// `P` is a permutation matrix. It is implemented through a pivoted QR
/// algorithm so that the forst `r` columns  of `QR` are a good approximation to the
/// `r` dominant columns of `A`. The matrix `R` has dimension `(k, m)` with
/// `k=min(m, n)`. `P` is of dimension `(n, n)`. The user can choose in the
/// method [MatrixQrDecomposition::get_q_alloc] how many columns of `Q` to return.
pub trait MatrixQrDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Array implementation
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;
    /// Create new
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>) -> RlstResult<Self>;

    /// Return the permuation vector for the QR decomposition.
    ///
    /// If `perm[i] = j` then the ith column of QR corresponds
    /// to the jth column of the original array.
    fn get_perm(&self) -> Vec<usize>;

    /// Return the R matrix of the QR decomposition.
    ///
    /// If `A` has dimension `(m, n)` then R has
    /// dimension `(k, n)` with `k=min(m, n)`.
    fn get_r<
        ArrayImplR: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplR, 2>,
    );

    /// Return the permuation matrix `P`.
    ///
    /// For `A` an `(m,n)` matrix `P` has dimension `(n, n)`.
    fn get_p<
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplQ, 2>,
    );

    /// Return the Q matrix of the QR decomposition.
    ///
    /// If `A` has dimension `(m, n)` then `arr` needs
    /// to be of dimension `(m, r)`, where `r<= m``
    /// is the desired number of columns of `Q`.
    ///
    /// This method allocates temporary memory during execution.
    fn get_q_alloc<
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplQ, 2>,
    ) -> RlstResult<()>;

    /// Apply Q to a given matrix.
    ///
    /// This method allocates temporary memory during execution.
    fn apply_q_alloc<
        ArrayImplQ: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + RawAccessMut<Item = Self::Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplQ, 2>,
        side: ApplyQSide,
        trans: ApplyQTrans,
    ) -> RlstResult<()>;
}

/// Transpose
#[derive(Clone, Copy)]
pub enum ApplyQTrans {
    /// No transpose
    NoTrans,
    /// Conjugate transpose
    ConjTrans,
}

/// QR decomposition
pub struct QrDecomposition<
    Item: RlstScalar,
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
            > MatrixQrDecomposition for QrDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;
            fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
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

            /// Return the permuation vector for the QR decomposition.
            ///
            /// If `perm[i] = j` then the ith column of QR corresponds
            /// to the jth column of the original array.
            fn get_perm(&self) -> Vec<usize> {
                self.jpvt
                    .iter()
                    .map(|&elem| elem as usize - 1)
                    .collect_vec()
            }

            /// Return the R matrix of the QR decomposition.
            ///
            /// If `A` has dimension `(m, n)` then R has
            /// dimension `(k, n)` with `k=min(m, n)`.
            fn get_r<
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

            /// Return the permuation matrix `P`.
            ///
            /// For `A` an `(m,n)` matrix `P` has dimension `(n, n)`.
            fn get_p<
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

            /// Return the Q matrix of the QR decomposition.
            ///
            /// If `A` has dimension `(m, n)` then `arr` needs
            /// to be of dimension `(m, r)`, where `r<= m``
            /// is the desired number of columns of `Q`.
            ///
            /// This method allocates temporary memory during execution.
            fn get_q_alloc<
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

            /// Apply Q to a given matrix.
            ///
            /// This method allocates temporary memory during execution.
            fn apply_q_alloc<
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
    };
}

macro_rules! implement_qr_complex {
    ($scalar:ty, $geqp3:expr, $ormqr:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixQrDecomposition for QrDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
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

                let mut rwork =
                    vec![<<$scalar as RlstScalar>::Real as Zero>::zero(); 2 * n as usize];

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

            /// Return the permuation vector for the QR decomposition.
            ///
            /// If `perm[i] = j` then the ith column of QR corresponds
            /// to the jth column of the original array.
            fn get_perm(&self) -> Vec<usize> {
                self.jpvt
                    .iter()
                    .map(|&elem| elem as usize - 1)
                    .collect_vec()
            }

            /// Return the R matrix of the QR decomposition.
            ///
            /// If `A` has dimension `(m, n)` then R has
            /// dimension `(k, n)` with `k=min(m, n)`.
            fn get_r<
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

            /// Return the permuation matrix `P`.
            ///
            /// For `A` an `(m,n)` matrix `P` has dimension `(n, n)`.
            fn get_p<
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

            /// Return the Q matrix of the QR decomposition.
            ///
            /// If `A` has dimension `(m, n)` then `arr` needs
            /// to be of dimension `(m, r)`, where `r<= m``
            /// is the desired number of columns of `Q`.
            ///
            /// This method allocates temporary memory during execution.
            fn get_q_alloc<
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

            /// Apply Q to a given matrix.
            ///
            /// This method allocates temporary memory during execution.
            fn apply_q_alloc<
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
    };
}

implement_qr_real!(f64, dgeqp3, dormqr);
implement_qr_real!(f32, sgeqp3, sormqr);
implement_qr_complex!(c64, zgeqp3, zunmqr);
implement_qr_complex!(c32, cgeqp3, cunmqr);
