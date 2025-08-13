//! QR Decomposition and its variants.
use super::assert_lapack_stride;
use crate::dense::array::Array;
use crate::dense::traits::{
    RandomAccessByValue, RandomAccessMut, RawAccess, RawAccessMut, Shape, Stride,
    UnsafeRandomAccessByValue, UnsafeRandomAccessMut,
};
use crate::dense::types::{c32, c64, RlstError, RlstResult, RlstScalar};
use crate::{
    empty_array, rlst_dynamic_array2, DynamicArray, GivensRotation, GivensRotationData,
    MultIntoResize, Side, TransMode, TriangularMatrix, TriangularOperations, TriangularType,
};
use itertools::Itertools;
use lapack::{
    cgeqp3, cgeqrf, cunmqr, dgeqp3, dgeqrf, dormqr, sgeqp3, sgeqrf, sormqr, zgeqp3, zgeqrf, zunmqr,
};
use num::Zero;
use serde::Deserialize;

/// Compute a QR decomposition from a given two-dimensional array.
pub trait MatrixQr: RlstScalar {
    /// Compute the matrix inverse
    fn into_qr_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        pivoting: Pivoting,
    ) -> RlstResult<QrDecomposition<Self, ArrayImpl>>;
}

macro_rules! implement_into_qr {
    ($scalar:ty) => {
        impl MatrixQr for $scalar {
            fn into_qr_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                pivoting: Pivoting,
            ) -> RlstResult<QrDecomposition<Self, ArrayImpl>> {
                QrDecomposition::<$scalar, ArrayImpl>::new(arr, pivoting)
            }
        }
    };
}

implement_into_qr!(f32);
implement_into_qr!(f64);
implement_into_qr!(c32);
implement_into_qr!(c64);

impl<
        Item: RlstScalar + MatrixQr,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the QR decomposition of a given 2-dimensional array.
    pub fn into_qr_alloc(self, pivoting: Pivoting) -> RlstResult<QrDecomposition<Item, ArrayImpl>> {
        <Item as MatrixQr>::into_qr_alloc(self, pivoting)
    }
}

/// Compute a RRQR decomposition from a given two-dimensional array.
pub trait MatrixRankRevealingQr: RlstScalar {
    /// Compute the matrix inverse
    fn into_rrqr_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + UnsafeRandomAccessMut<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
        qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
        rank_param: RankParam<<Self as RlstScalar>::Real>,
    ) -> RankRevealingQrDecomposition<Self>;
}

macro_rules! implement_into_rrqr {
    ($scalar:ty) => {
        impl MatrixRankRevealingQr for $scalar {
            fn into_rrqr_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + UnsafeRandomAccessMut<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
                qr_type: RankRevealingQrType<<Self as RlstScalar>::Real>,
                rank_param: RankParam<<Self as RlstScalar>::Real>,
            ) -> RankRevealingQrDecomposition<Self> {
                RankRevealingQrDecomposition::<$scalar>::new(arr, qr_type, rank_param)
            }
        }
    };
}

implement_into_rrqr!(f32);
implement_into_rrqr!(f64);
implement_into_rrqr!(c32);
implement_into_rrqr!(c64);

impl<
        Item: RlstScalar + MatrixQr + MatrixRankRevealingQr,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + UnsafeRandomAccessMut<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the QR decomposition of a given 2-dimensional array.
    pub fn into_rrqr_alloc(
        self,
        qr_type: RankRevealingQrType<<Item as RlstScalar>::Real>,
        rank_param: RankParam<<Item as RlstScalar>::Real>,
    ) -> RankRevealingQrDecomposition<Item> {
        <Item as MatrixRankRevealingQr>::into_rrqr_alloc(self, qr_type, rank_param)
    }
}

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

/// QR decomposition
pub struct QrDecomposition<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    tau: Vec<Item>,
    jpvt: Vec<i32>,
}

/// RankRevealing QR decomposition
pub struct RankRevealingQrDecomposition<Item: RlstScalar> {
    /// Factor Q
    pub q: DynamicArray<Item, 2>,
    /// Factor R
    pub r: DynamicArray<Item, 2>,
    /// Permutation induced
    pub perm: Vec<usize>,
    /// Associated rank
    pub rank: usize,
}

#[derive(Debug, Clone, Deserialize)]
/// Define if QR is performed with or without pivoting
pub enum RankRevealingQrType<Item> {
    /// Rank-Revealing QR
    RRQR,
    /// Strong Rank-Revealing QR
    SRRQR(Item),
}

/// Regular or Pivoted QR
pub enum Pivoting {
    /// Pivoting
    True,
    /// No Pivoting
    False,
}

#[derive(Clone)]
/// Tolerance used to compute the rank
pub enum QrTolerance {
    /// Absolute
    Abs,
    /// Relative
    Rel,
}

/// Strategy to compute the rank
pub enum RankParam<Item> {
    /// Tolerance and tolerance type (absolute or relative)
    Tol(Item, QrTolerance),
    /// Rank
    Rank(usize),
}

/// Compute the QR decomposition of a matrix.
///
/// The QR Decomposition of an `(m,n)` matrix `A` is defined
/// by `AP = QR`, where `P` is an `(n, n)` permutation matrix,
/// `Q` is a `(m, m)` orthogonal matrix, and `R` is
/// an `(m, n)` upper triangular matrix.
pub trait MatrixQrDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Array implementaion
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;

    /// Create a new QR Decomposition.
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>, pivoting: Pivoting) -> RlstResult<Self>;

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

/// Trait for Rank-Revealing QR Decompositions
///
/// RRQR returns a Pivoted QR and a rank for a certain tolerance tol (or a fixed rank)
/// SRRQR returns a more stable RRQR and a rank for a certain f, and a tolerance tol (or a fixed rank)
pub trait RankRevealingMatrixQrDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;

    /// Create a new QR Decomposition.
    fn new<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        qr_type: RankRevealingQrType<<Self::Item as RlstScalar>::Real>,
        rank_param: RankParam<<Self::Item as RlstScalar>::Real>,
    ) -> Self;

    /// RRQR returns a Pivoted QR and a rank for a certain tolerance tol (or a fixed rank)
    fn rrqr<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        rank_param: RankParam<<Self::Item as RlstScalar>::Real>,
    ) -> (
        DynamicArray<Self::Item, 2>,
        DynamicArray<Self::Item, 2>,
        Vec<usize>,
        usize,
    );

    /// SRRQR returns a more stable RRQR and a rank for a certain f
    fn srrqr<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        rank_param: RankParam<<Self::Item as RlstScalar>::Real>,
        f: <Self::Item as RlstScalar>::Real,
    ) -> (
        DynamicArray<Self::Item, 2>,
        DynamicArray<Self::Item, 2>,
        Vec<usize>,
        usize,
    );

    /// SRRQR returns a more stable RRQR and a rank for a fixed rank
    fn srrqr_k<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        f: <Self::Item as RlstScalar>::Real,
        rank: usize,
    ) -> (
        DynamicArray<Self::Item, 2>,
        DynamicArray<Self::Item, 2>,
        Vec<usize>,
        usize,
    );

    /// SRRQR returns a more stable RRQR and a rank for a certain f, and a tolerance tol
    fn srrqr_tol<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self::Item>,
    >(
        arr: Array<Self::Item, ArrayImplMut, 2>,
        f: <Self::Item as RlstScalar>::Real,
        tol: <Self::Item as RlstScalar>::Real,
        tol_type: QrTolerance,
    ) -> (
        DynamicArray<Self::Item, 2>,
        DynamicArray<Self::Item, 2>,
        Vec<usize>,
        usize,
    );
}

macro_rules! implement_qr_real {
    ($scalar:ty, $geqp3:expr, $geqrf: expr, $ormqr:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixQrDecomposition for QrDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            fn new(mut arr: Array<$scalar, ArrayImpl, 2>, pivoting: Pivoting) -> RlstResult<Self> {
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
                    match pivoting {
                        Pivoting::True => $geqp3(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut jpvt,
                            &mut tau,
                            &mut work_query,
                            lwork,
                            &mut info,
                        ),
                        Pivoting::False => $geqrf(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut tau,
                            &mut work_query,
                            lwork,
                            &mut info,
                        ),
                    };
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    match pivoting {
                        Pivoting::True => $geqp3(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut jpvt,
                            &mut tau,
                            &mut work,
                            lwork,
                            &mut info,
                        ),
                        Pivoting::False => $geqrf(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut tau,
                            &mut work,
                            lwork,
                            &mut info,
                        ),
                    };
                }

                match info {
                    0 => Ok(Self { arr, tau, jpvt }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }

            fn get_perm(&self) -> Vec<usize> {
                self.jpvt
                    .iter()
                    .map(|&elem| elem as usize - 1)
                    .collect_vec()
            }

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
    ($scalar:ty, $geqp3:expr, $geqrf: expr, $ormqr:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixQrDecomposition for QrDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;
            /// Create a new QR Decomposition.
            fn new(mut arr: Array<$scalar, ArrayImpl, 2>, pivoting: Pivoting) -> RlstResult<Self> {
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
                    match pivoting {
                        Pivoting::True => $geqp3(
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
                        ),
                        Pivoting::False => $geqrf(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut tau,
                            &mut work_query,
                            lwork,
                            &mut info,
                        ),
                    };
                }

                match info {
                    0 => (),
                    _ => return Err(RlstError::LapackError(info)),
                }

                let lwork = work_query[0].re() as i32;
                let mut work = vec![<$scalar as Zero>::zero(); lwork as usize];

                unsafe {
                    match pivoting {
                        Pivoting::True => $geqp3(
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
                        ),
                        Pivoting::False => $geqrf(
                            m,
                            n,
                            arr.data_mut(),
                            lda,
                            &mut tau,
                            &mut work,
                            lwork,
                            &mut info,
                        ),
                    };
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

implement_qr_real!(f64, dgeqp3, dgeqrf, dormqr);
implement_qr_real!(f32, sgeqp3, sgeqrf, sormqr);
implement_qr_complex!(c64, zgeqp3, zgeqrf, zunmqr);
implement_qr_complex!(c32, cgeqp3, cgeqrf, cunmqr);

macro_rules! implement_special_qr_real {
    ($scalar:ty, $geqp3:expr, $geqrf: expr, $ormqr:expr) => {
        impl RankRevealingMatrixQrDecomposition for RankRevealingQrDecomposition<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImplMut, 2>,
                qr_type: RankRevealingQrType<<$scalar as RlstScalar>::Real>,
                rank_param: RankParam<<$scalar as RlstScalar>::Real>,
            ) -> Self {
                let (q, r, perm, rank) = match qr_type {
                    RankRevealingQrType::RRQR => Self::rrqr(arr, rank_param),
                    RankRevealingQrType::SRRQR(f) => Self::srrqr(arr, rank_param, f),
                };

                Self { q, r, perm, rank }
            }

            fn srrqr<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImplMut, 2>,
                rank_param: RankParam<<$scalar as RlstScalar>::Real>,
                f: <$scalar as RlstScalar>::Real,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                match rank_param {
                    RankParam::Tol(tol, tol_type) => Self::srrqr_tol(arr, f, tol, tol_type),
                    RankParam::Rank(rank) => Self::srrqr_k(arr, f, rank),
                }
            }

            fn rrqr<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut arr: Array<$scalar, ArrayImplMut, 2>,
                rank_param: RankParam<<$scalar as RlstScalar>::Real>,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                let [m, n] = arr.shape();
                let qr = arr.r_mut().into_qr_alloc(Pivoting::True).unwrap();
                let perm = qr.get_perm();

                let (mut r, mut q) = if m < n {
                    let mut r: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [m, n]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, m]);
                    qr.get_r(r.r_mut());
                    let _ = qr.get_q_alloc(q.r_mut());
                    (r, q)
                } else {
                    let mut r: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [n, n]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, n]);
                    qr.get_r(r.r_mut());
                    let _ = qr.get_q_alloc(q.r_mut());
                    (r, q)
                };

                let rank = match rank_param {
                    RankParam::Tol(tol, tol_type) => match tol_type {
                        QrTolerance::Abs => rank_from_tolerance_abs(&mut r, &mut q, tol),
                        QrTolerance::Rel => rank_from_tolerance_rel(&mut r, &mut q, tol),
                    },
                    RankParam::Rank(rank) => rank,
                };

                (q, r, perm, rank)
            }

            fn srrqr_k<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut arr: Array<$scalar, ArrayImplMut, 2>,
                mut f: <$scalar as RlstScalar>::Real,
                rank: usize,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                fn get_r_blocks(
                    r: &DynamicArray<$scalar, 2>,
                    dim: usize,
                    rank: usize,
                ) -> (
                    TriangularMatrix<$scalar>,
                    DynamicArray<$scalar, 2>,
                    DynamicArray<$scalar, 2>,
                ) {
                    let r11 = TriangularMatrix::<$scalar>::new(
                        &r.r().into_subview([0, 0], [rank, rank]),
                        TriangularType::Upper,
                    )
                    .unwrap();
                    let mut r22 = empty_array();
                    r22.r_mut().fill_from_resize(
                        r.r().into_subview([rank, rank], [dim - rank, dim - rank]),
                    );

                    let mut r12 = empty_array();
                    r12.r_mut()
                        .fill_from_resize(r.r().into_subview([0, rank], [rank, dim - rank]));

                    (r11, r12, r22)
                }

                fn find_indices(
                    id_mat: &DynamicArray<$scalar, 2>,
                    gamma: &[$scalar],
                    omega: &[$scalar],
                    f: <$scalar as RlstScalar>::Real,
                ) -> Option<(usize, usize)> {
                    let shape = id_mat.shape();
                    let f2 = f.powi(2);
                    for j in 0..shape[1] {
                        for i in 0..shape[0] {
                            let remaining_blocks_norm = (gamma[j] * (1.0 / omega[i])).abs().powi(2);
                            let id_mat_norm = id_mat[[i, j]].abs().powi(2);

                            if id_mat_norm + remaining_blocks_norm > f2 {
                                return Some((i, j));
                            }
                        }
                    }
                    None
                }

                fn permute_matrix(
                    mat: &mut DynamicArray<$scalar, 2>,
                    i: usize,
                    j: usize,
                    num_rows: Option<usize>,
                ) {
                    let num_rows = match num_rows {
                        Some(row) => row,
                        None => mat.shape()[0],
                    };

                    for row in 0..num_rows {
                        let temp = mat.r()[[row, i]];
                        mat.r_mut()[[row, i]] = mat.r()[[row, j]];
                        mat.r_mut()[[row, j]] = temp;
                    }
                }

                if f < 1.0 {
                    println!("Parameter f given is less than 1. Automatically set f = 2");
                    f = 2.0;
                }

                let (mut q, mut r, mut perm, rank) = Self::rrqr(arr.r_mut(), RankParam::Rank(rank));
                let [m, n] = arr.shape();
                let r_shape = r.shape();
                let q_shape = q.shape();

                let dim = m.min(n);

                let (r11, mut r12, r22) = get_r_blocks(&r, dim, rank);
                let mut r11_inv = rlst_dynamic_array2!($scalar, [rank, rank]);
                r11_inv.r_mut().set_identity();
                r11.solve(&mut r11_inv, Side::Left, TransMode::NoTrans);
                r11.solve(&mut r12, Side::Left, TransMode::NoTrans);

                let mut gamma = (0..dim - rank)
                    .map(|j| Self::Item::from(r22.r().slice(1, j).norm_2()))
                    .collect::<Vec<$scalar>>();

                let mut omega = (0..rank)
                    .map(|i| Self::Item::from(1.0 / r11_inv.r().slice(0, i).norm_2()))
                    .collect::<Vec<$scalar>>();

                loop {
                    let indices = find_indices(&r12, &gamma, &omega, f);

                    let (i, j) = match indices {
                        Some(inds) => inds,
                        None => break,
                    };

                    if j > 0 {
                        permute_matrix(&mut r12, 0, j, None);
                        gamma.swap(0, j);
                        permute_matrix(&mut r, rank, rank + j, None);
                        perm.swap(rank, rank + j);
                    }

                    // Second step : interchanging the i and k th columns
                    if i < rank - 1 {
                        let mut order: Vec<usize> = (i + 1..rank).collect(); // rank is exclusive here
                        order.push(i);

                        // Copy the original values before writing back
                        let original = perm.clone();

                        for (dest, src) in (i..rank).zip(order.iter().copied()) {
                            perm[dest] = original[src];
                            omega[dest] = omega[src];
                        }

                        let mut original_r = empty_array(); // Copy original before modifying
                        original_r.r_mut().fill_from_resize(r.r());
                        for row in 0..r_shape[0] {
                            for (dest, src) in (i..rank).zip(order.iter().copied()) {
                                r.r_mut()[[row, dest]] = original_r[[row, src]];
                            }
                        }

                        // --- Row permutation for AB(i:k, :) ---
                        let mut original_r12 = empty_array(); // Copy original before modifying
                        original_r12.r_mut().fill_from_resize(r12.r());
                        let r12_shape = r12.shape();
                        for (dest, src) in (i..rank).zip(order.iter().copied()) {
                            for col in 0..r12_shape[1] {
                                r12.r_mut()[[dest, col]] = original_r12[[src, col]];
                            }
                        }
                        // Givens rotation for the triangulation of R(1:k, 1:k)
                        for ind in i..rank - 1 {
                            let mut r_block = r.r_mut().into_subview([ind, 0], [2, r_shape[1]]);
                            let vec_r = r_block.r().into_subview([0, ind], [2, 1]);
                            let mut givens_rotation: GivensRotationData<$scalar> =
                                GivensRotation::<$scalar>::new(vec_r[[0, 0]], vec_r[[1, 0]]);
                            let tmp = givens_rotation.c * vec_r[[0, 0]]
                                + givens_rotation.s * vec_r[[1, 0]];

                            if tmp < <$scalar as num::Zero>::zero() {
                                // guarantee r(ind, ind) > 0
                                givens_rotation.c = -givens_rotation.c;
                                givens_rotation.s = -givens_rotation.s;
                                givens_rotation.r = -givens_rotation.r;
                            }

                            let g = givens_rotation.get_givens_matrix();
                            let mut rg: DynamicArray<$scalar, 2> = empty_array();
                            rg.r_mut().simple_mult_into_resize(g.r(), r_block.r());
                            r_block.r_mut().fill_from(rg.r());

                            let mut q_block = q.r_mut().into_subview([0, ind], [q_shape[0], 2]);
                            let mut gq: DynamicArray<$scalar, 2> = empty_array();
                            gq.r_mut().mult_into_resize(
                                TransMode::NoTrans,
                                TransMode::ConjTrans,
                                num::One::one(),
                                q_block.r(),
                                g.r(),
                                num::Zero::zero(),
                            );
                            q_block.r_mut().fill_from(gq.r());
                        }
                        if r.r()[[rank - 1, rank - 1]] < <$scalar as num::Zero>::zero() {
                            r.r_mut()
                                .into_subview([rank - 1, 0], [1, r_shape[1]])
                                .scale_inplace(
                                    <$scalar as num::Zero>::zero() - <$scalar as num::One>::one(),
                                );
                            q.r_mut()
                                .into_subview([0, rank - 1], [r_shape[0], 1])
                                .scale_inplace(
                                    <$scalar as num::Zero>::zero() - <$scalar as num::One>::one(),
                                );
                        }
                    }

                    // Third step : zeroing out the below-diag of k+1 th columns
                    if rank - 1 < r_shape[0] {
                        for ind in (rank + 1)..r_shape[0] {
                            let vec_r = [r.r()[[rank, rank]], r.r()[[ind, rank]]];
                            let mut givens_rotation: GivensRotationData<$scalar> =
                                GivensRotation::<$scalar>::new(vec_r[0], vec_r[1]);
                            let tmp = givens_rotation.c * vec_r[0] + givens_rotation.s * vec_r[1];
                            if tmp < <$scalar as num::Zero>::zero() {
                                givens_rotation.c = -givens_rotation.c;
                                givens_rotation.s = -givens_rotation.s;
                                givens_rotation.r = -givens_rotation.r;
                            }
                            let g = givens_rotation.get_givens_matrix();
                            let mut r_block = rlst_dynamic_array2!($scalar, [2, r_shape[1]]);
                            r_block
                                .r_mut()
                                .into_subview([0, 0], [1, r_shape[1]])
                                .fill_from(r.r().into_subview([rank, 0], [1, r_shape[1]]));
                            r_block
                                .r_mut()
                                .into_subview([1, 0], [1, r_shape[1]])
                                .fill_from(r.r().into_subview([ind, 0], [1, r_shape[1]]));
                            let mut rg: DynamicArray<$scalar, 2> = empty_array();
                            rg.r_mut().simple_mult_into_resize(g.r(), r_block.r());
                            r.r_mut()
                                .into_subview([rank, 0], [1, r_shape[1]])
                                .fill_from(rg.r().into_subview([0, 0], [1, r_shape[1]]));
                            r.r_mut()
                                .into_subview([ind, 0], [1, r_shape[1]])
                                .fill_from(rg.r().into_subview([1, 0], [1, r_shape[1]]));

                            let mut q_block = rlst_dynamic_array2!($scalar, [q_shape[0], 2]);
                            q_block
                                .r_mut()
                                .into_subview([0, 0], [q_shape[0], 1])
                                .fill_from(q.r().into_subview([0, rank], [q_shape[0], 1]));
                            q_block
                                .r_mut()
                                .into_subview([0, 1], [q_shape[0], 1])
                                .fill_from(q.r().into_subview([0, ind], [q_shape[0], 1]));
                            let mut gq: DynamicArray<$scalar, 2> = empty_array();
                            gq.r_mut().mult_into_resize(
                                TransMode::NoTrans,
                                TransMode::ConjTrans,
                                num::One::one(),
                                q_block.r(),
                                g.r(),
                                num::Zero::zero(),
                            );
                            q.r_mut()
                                .into_subview([0, rank], [q_shape[0], 1])
                                .fill_from(gq.r().into_subview([0, 0], [q_shape[0], 1]));
                            q.r_mut()
                                .into_subview([0, ind], [q_shape[0], 1])
                                .fill_from(gq.r().into_subview([0, 1], [q_shape[0], 1]));
                        }
                    }

                    perm.swap(rank, rank - 1);
                    // Fourth step : interchaing the k and k+1 th columns
                    let ga = r.r()[[rank - 1, rank - 1]];
                    let mu = r.r()[[rank - 1, rank]] / ga;
                    let nu = if rank < r_shape[0] {
                        r.r()[[rank, rank]] / ga
                    } else {
                        <$scalar as num::Zero>::zero()
                    };

                    let mut p_mat = rlst_dynamic_array2!($scalar, [2, 2]);
                    p_mat.r_mut()[[0, 0]] = mu;
                    p_mat.r_mut()[[0, 1]] = nu;
                    p_mat.r_mut()[[1, 0]] = nu;
                    p_mat.r_mut()[[1, 1]] = -mu;

                    let rho = (mu * mu + nu * nu).sqrt();
                    let ga_bar = ga * rho;

                    let mut ct = empty_array();
                    ct.r_mut().fill_from_resize(
                        r.r()
                            .into_subview([rank - 1, rank + 1], [2, r_shape[1] - (rank + 1)]),
                    );

                    let mut ct_bar = empty_array();
                    ct_bar.r_mut().simple_mult_into_resize(p_mat.r(), ct.r());
                    ct_bar.scale_inplace(1.0 / rho);
                    let ct_bar_shape = ct_bar.shape();

                    let mut u = empty_array();
                    u.r_mut()
                        .fill_from_resize(r.r().into_subview([0, rank - 1], [rank - 1, 1]));

                    // Modify R
                    permute_matrix(&mut r, rank - 1, rank, Some(rank - 1));
                    r.r_mut()[[rank - 1, rank - 1]] = ga_bar;
                    r.r_mut()[[rank - 1, rank]] = ga * mu / rho;
                    r.r_mut()[[rank, rank]] = ga * nu / rho;
                    r.r_mut()
                        .into_subview([rank - 1, rank + 1], [2, r_shape[1] - (rank + 1)])
                        .fill_from(ct_bar.r());

                    let r11_tmp = TriangularMatrix::<$scalar>::new(
                        &r.r().into_subview([0, 0], [rank - 1, rank - 1]),
                        TriangularType::Upper,
                    )
                    .unwrap();

                    let mut u1 = empty_array();
                    let r12_shape = r12.shape();

                    r11_tmp.solve(&mut u, Side::Left, TransMode::NoTrans);
                    u1.r_mut()
                        .fill_from_resize(r12.r().into_subview([0, 0], [rank - 1, 1]));

                    r12.r_mut()
                        .into_subview([0, 0], [rank - 1, 1])
                        .fill_from(((nu * nu) / (rho * rho)) * u.r() - (mu / (rho * rho)) * u1.r());
                    r12.r_mut()[[rank - 1, 0]] = mu / (rho * rho);
                    r12.r_mut()
                        .into_subview([rank - 1, 1], [1, r12_shape[1] - 1])
                        .fill_from(ct_bar.r().into_subview([0, 0], [1, ct_bar_shape[1]]));
                    r12.r_mut()
                        .into_subview([rank - 1, 1], [1, r12_shape[1] - 1])
                        .scale_inplace(1.0 / ga_bar);

                    let mut tmp1: DynamicArray<$scalar, 2> = empty_array();
                    let mut tmp2: DynamicArray<$scalar, 2> = empty_array();
                    let mut tmp3: DynamicArray<$scalar, 2> = empty_array();

                    tmp1.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        TransMode::NoTrans,
                        nu / ga_bar,
                        u.r(),
                        ct_bar.r().into_subview([1, 0], [1, ct_bar_shape[1]]),
                        num::Zero::zero(),
                    );

                    tmp2.r_mut().mult_into_resize(
                        TransMode::NoTrans,
                        TransMode::NoTrans,
                        -1.0 / ga_bar,
                        u1.r(),
                        ct_bar.r().into_subview([0, 0], [1, ct_bar_shape[1]]),
                        num::Zero::zero(),
                    );

                    tmp3.r_mut().fill_from_resize(
                        r12.r().into_subview([0, 1], [rank - 1, r12_shape[1] - 1]),
                    );
                    r12.r_mut()
                        .into_subview([0, 1], [rank - 1, r12_shape[1] - 1])
                        .fill_from(tmp1.r() + tmp2.r() + tmp3.r());

                    gamma[0] = ga * nu / rho;

                    for i in 1..gamma.len() {
                        gamma[i] = (gamma[i].powi(2) + ct_bar.r().slice(0, 1)[[i - 1]].powi(2)
                            - ct.r().slice(0, 1)[[i - 1]].powi(2))
                        .sqrt();
                    }

                    let mut u_bar = empty_array();
                    u_bar.r_mut().fill_from_resize(u1.r() + mu * u.r());

                    omega[rank - 1] = ga_bar;

                    for i in 0..(rank - 1) {
                        let val = omega[i].powi(-2)
                            + u_bar.r().data()[i].powi(2) / (ga_bar * ga_bar)
                            - u.r().data()[i].powi(2) / (ga * ga);
                        omega[i] = 1.0 / val.sqrt();
                    }

                    let givens_rotation = GivensRotationData {
                        c: -mu / rho,
                        s: -nu / rho,
                        r: num::One::one(),
                    };

                    if rank - 1 < r_shape[0] {
                        let g = givens_rotation.get_givens_matrix();
                        let q_shape = q.shape();
                        let mut q_block = q.r_mut().into_subview([0, rank - 1], [q_shape[0], 2]);
                        let mut gq: DynamicArray<$scalar, 2> = empty_array();
                        gq.r_mut().mult_into_resize(
                            TransMode::NoTrans,
                            TransMode::ConjTrans,
                            num::One::one(),
                            q_block.r(),
                            g.r(),
                            num::Zero::zero(),
                        );
                        q_block.r_mut().fill_from(gq.r());
                    }
                }

                let mut q_trunc = empty_array();
                let mut r_trunc = empty_array();
                q_trunc
                    .r_mut()
                    .fill_from_resize(q.r().into_subview([0, 0], [q_shape[0], rank]));
                r_trunc
                    .r_mut()
                    .fill_from_resize(r.r().into_subview([0, 0], [rank, r_shape[1]]));

                (q_trunc, r_trunc, perm, rank)
            }

            fn srrqr_tol<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut arr: Array<$scalar, ArrayImplMut, 2>,
                mut f: $scalar,
                tol: $scalar,
                tol_type: QrTolerance,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                fn get_r_blocks(
                    r: &DynamicArray<$scalar, 2>,
                    dim: usize,
                    rank: usize,
                ) -> (
                    TriangularMatrix<$scalar>,
                    DynamicArray<$scalar, 2>,
                    DynamicArray<$scalar, 2>,
                ) {
                    let r11 = TriangularMatrix::<$scalar>::new(
                        &r.r().into_subview([0, 0], [rank, rank]),
                        TriangularType::Upper,
                    )
                    .unwrap();
                    let mut r22 = empty_array();
                    r22.r_mut().fill_from_resize(
                        r.r().into_subview([rank, rank], [dim - rank, dim - rank]),
                    );

                    let mut r12 = empty_array();
                    r12.r_mut()
                        .fill_from_resize(r.r().into_subview([0, rank], [rank, dim - rank]));

                    (r11, r12, r22)
                }

                fn find_indices(
                    id_mat: &DynamicArray<$scalar, 2>,
                    gamma: &[$scalar],
                    omega: &[$scalar],
                    f: <$scalar as RlstScalar>::Real,
                ) -> Option<(usize, usize)> {
                    let shape = id_mat.shape();
                    let f2 = f.pow(2.0);
                    for j in 0..shape[1] {
                        for i in 0..shape[0] {
                            let remaining_blocks_norm = (gamma[j] * (1.0 / omega[i])).pow(2.0);
                            let id_mat_norm = id_mat[[i, j]].abs().pow(2.0);

                            if id_mat_norm + remaining_blocks_norm > f2 {
                                return Some((i, j));
                            }
                        }
                    }
                    None
                }

                fn permute_matrix(
                    mat: &mut DynamicArray<$scalar, 2>,
                    i: usize,
                    j: usize,
                    num_rows: Option<usize>,
                ) {
                    let num_rows = match num_rows {
                        Some(row) => row,
                        None => mat.shape()[0],
                    };

                    for row in 0..num_rows {
                        let temp = mat.r()[[row, i]];
                        mat.r_mut()[[row, i]] = mat.r()[[row, j]];
                        mat.r_mut()[[row, j]] = temp;
                    }
                }

                println!("1");

                if f < 1.0 {
                    println!("Parameter f given is less than 1. Automatically set f = 2");
                    f = 2.0;
                }
                println!("2");
                let (mut q, mut r, mut perm, mut rank) =
                    Self::rrqr(arr.r_mut(), RankParam::Tol(tol, tol_type.clone()));
                let [m, n] = arr.shape();
                let r_shape = r.shape();
                let q_shape = q.shape();

                if rank == r_shape[0] {
                    return (q, r, perm, rank);
                }

                if rank > 1 {
                    let dim = m.min(n);
                    println!("3");
                    //let mut rank = rank_from_tolerance2(&mut r, &mut q, tol);
                    let (mut r11, mut r12, mut r22) = get_r_blocks(&r, dim, rank);
                    let mut r11_inv = rlst_dynamic_array2!($scalar, [rank, rank]);
                    r11_inv.r_mut().set_identity();
                    r11.solve(&mut r11_inv, Side::Left, TransMode::NoTrans);
                    r11.solve(&mut r12, Side::Left, TransMode::NoTrans);
                    println!("4");
                    let mut gamma = (0..dim - rank)
                        .map(|j| r22.r().slice(1, j).norm_2())
                        .collect::<Vec<_>>();

                    let mut omega = (0..rank)
                        .map(|i| 1.0 / r11_inv.r().slice(0, i).norm_2())
                        .collect::<Vec<_>>();
                    println!("5");
                    loop {
                        loop {
                            println!("6");
                            let indices = find_indices(&r12, &gamma, &omega, f);

                            let (i, j) = match indices {
                                Some(inds) => inds,
                                None => break,
                            };
                            println!("7");
                            if j > 0 {
                                permute_matrix(&mut r12, 0, j, None);
                                gamma.swap(0, j);
                                permute_matrix(&mut r, rank, rank + j, None);
                                perm.swap(rank, rank + j);
                            }
                            println!("8");
                            // Second step : interchanging the i and k th columns
                            if i < rank - 1 {
                                println!("9");
                                let mut order: Vec<usize> = (i + 1..rank).collect(); // rank is exclusive here
                                order.push(i);

                                // Copy the original values before writing back
                                let original = perm.clone();

                                for (dest, src) in (i..rank).zip(order.iter().copied()) {
                                    perm[dest] = original[src];
                                    omega[dest] = omega[src];
                                }

                                let mut original_r = empty_array(); // Copy original before modifying
                                original_r.r_mut().fill_from_resize(r.r());
                                for row in 0..r_shape[0] {
                                    for (dest, src) in (i..rank).zip(order.iter().copied()) {
                                        r.r_mut()[[row, dest]] = original_r[[row, src]];
                                    }
                                }

                                // --- Row permutation for AB(i:k, :) ---
                                let mut original_r12 = empty_array(); // Copy original before modifying
                                original_r12.r_mut().fill_from_resize(r12.r());
                                let r12_shape = r12.shape();
                                for (dest, src) in (i..rank).zip(order.iter().copied()) {
                                    for col in 0..r12_shape[1] {
                                        r12.r_mut()[[dest, col]] = original_r12[[src, col]];
                                    }
                                }
                                println!("10");
                                // Givens rotation for the triangulation of R(1:k, 1:k)
                                for ind in i..rank {
                                    println!("11");
                                    let mut r_block =
                                        r.r_mut().into_subview([ind, 0], [2, r_shape[1]]);
                                    let vec_r = r_block.r().into_subview([0, ind], [2, 1]);
                                    let mut givens_rotation: GivensRotationData<$scalar> =
                                        GivensRotation::<$scalar>::new(
                                            vec_r[[0, 0]],
                                            vec_r[[1, 0]],
                                        );
                                    let tmp = givens_rotation.c * vec_r[[0, 0]]
                                        + givens_rotation.s * vec_r[[1, 0]];
                                    println!("12");
                                    if tmp < 0.0 {
                                        // guarantee r(ind, ind) > 0
                                        givens_rotation.c = -givens_rotation.c;
                                        givens_rotation.s = -givens_rotation.s;
                                        givens_rotation.r = -givens_rotation.r;
                                    }
                                    println!("13");
                                    let g = givens_rotation.get_givens_matrix();
                                    let mut rg: DynamicArray<$scalar, 2> = empty_array();
                                    rg.r_mut().simple_mult_into_resize(g.r(), r_block.r());
                                    r_block.r_mut().fill_from(rg.r());
                                    println!("14");
                                    let mut q_block =
                                        q.r_mut().into_subview([0, ind], [q_shape[0], 2]);
                                    let mut gq: DynamicArray<$scalar, 2> = empty_array();
                                    gq.r_mut().mult_into_resize(
                                        TransMode::NoTrans,
                                        TransMode::ConjTrans,
                                        num::One::one(),
                                        q_block.r(),
                                        g.r(),
                                        num::Zero::zero(),
                                    );
                                    q_block.r_mut().fill_from(gq.r());
                                    println!("15");
                                }
                                println!("16");
                                if r.r()[[rank - 1, rank - 1]] < 0.0 {
                                    r.r_mut()
                                        .into_subview([rank - 1, 0], [1, r_shape[1]])
                                        .scale_inplace(-1.0);
                                    q.r_mut()
                                        .into_subview([0, rank - 1], [r_shape[0], 1])
                                        .scale_inplace(-1.0);
                                }
                                println!("17");
                            }

                            // Third step : zeroing out the below-diag of k+1 th columns
                            if rank < r_shape[0] {
                                println!("18.0, {:?}, {}", r_shape, rank - 1);
                                for ind in (rank + 1)..r_shape[0] {
                                    println!("18, {:?}", r_shape);
                                    let vec_r = [r.r()[[rank, rank]], r.r()[[ind, rank]]];
                                    let mut givens_rotation: GivensRotationData<$scalar> =
                                        GivensRotation::<$scalar>::new(vec_r[0], vec_r[1]);
                                    let tmp =
                                        givens_rotation.c * vec_r[0] + givens_rotation.s * vec_r[1];
                                    if tmp < 0.0 {
                                        givens_rotation.c = -givens_rotation.c;
                                        givens_rotation.s = -givens_rotation.s;
                                        givens_rotation.r = -givens_rotation.r;
                                    }
                                    println!("19");
                                    let g = givens_rotation.get_givens_matrix();
                                    let mut r_block =
                                        rlst_dynamic_array2!($scalar, [2, r_shape[1]]);
                                    r_block
                                        .r_mut()
                                        .into_subview([0, 0], [1, r_shape[1]])
                                        .fill_from(r.r().into_subview([rank, 0], [1, r_shape[1]]));
                                    r_block
                                        .r_mut()
                                        .into_subview([1, 0], [1, r_shape[1]])
                                        .fill_from(r.r().into_subview([ind, 0], [1, r_shape[1]]));
                                    println!("20");
                                    let mut rg: DynamicArray<$scalar, 2> = empty_array();
                                    rg.r_mut().simple_mult_into_resize(g.r(), r_block.r());
                                    r.r_mut()
                                        .into_subview([rank, 0], [1, r_shape[1]])
                                        .fill_from(rg.r().into_subview([0, 0], [1, r_shape[1]]));
                                    r.r_mut()
                                        .into_subview([ind, 0], [1, r_shape[1]])
                                        .fill_from(rg.r().into_subview([1, 0], [1, r_shape[1]]));
                                    println!("21");
                                    let mut q_block =
                                        rlst_dynamic_array2!($scalar, [q_shape[0], 2]);
                                    q_block
                                        .r_mut()
                                        .into_subview([0, 0], [q_shape[0], 1])
                                        .fill_from(q.r().into_subview([0, rank], [q_shape[0], 1]));
                                    q_block
                                        .r_mut()
                                        .into_subview([0, 1], [q_shape[0], 1])
                                        .fill_from(q.r().into_subview([0, ind], [q_shape[0], 1]));
                                    println!("22");
                                    let mut gq: DynamicArray<$scalar, 2> = empty_array();
                                    gq.r_mut().mult_into_resize(
                                        TransMode::NoTrans,
                                        TransMode::ConjTrans,
                                        num::One::one(),
                                        q_block.r(),
                                        g.r(),
                                        num::Zero::zero(),
                                    );
                                    println!("23");
                                    q.r_mut()
                                        .into_subview([0, rank], [q_shape[0], 1])
                                        .fill_from(gq.r().into_subview([0, 0], [q_shape[0], 1]));
                                    q.r_mut()
                                        .into_subview([0, ind], [q_shape[0], 1])
                                        .fill_from(gq.r().into_subview([0, 1], [q_shape[0], 1]));
                                    println!("24");
                                }
                            }

                            println!("25");
                            perm.swap(rank, rank - 1);
                            // Fourth step : interchaing the k and k+1 th columns
                            let ga = r.r()[[rank - 1, rank - 1]];
                            let mu = r.r()[[rank - 1, rank]] / ga;
                            let nu = if rank < r_shape[0] {
                                r.r()[[rank, rank]] / ga
                            } else {
                                0.0
                            };
                            println!("26");
                            let mut p_mat = rlst_dynamic_array2!($scalar, [2, 2]);
                            p_mat.r_mut()[[0, 0]] = mu;
                            p_mat.r_mut()[[0, 1]] = nu;
                            p_mat.r_mut()[[1, 0]] = nu;
                            p_mat.r_mut()[[1, 1]] = -mu;
                            println!("27, {:?}, {}", r_shape, rank);
                            let rho = (mu * mu + nu * nu).sqrt();
                            let ga_bar = ga * rho;

                            let mut ct = empty_array();
                            ct.r_mut().fill_from_resize(
                                r.r().into_subview(
                                    [rank - 1, rank + 1],
                                    [2, r_shape[1] - (rank + 1)],
                                ),
                            );
                            println!("28");
                            let mut ct_bar = empty_array();
                            ct_bar.r_mut().simple_mult_into_resize(p_mat.r(), ct.r());
                            ct_bar.scale_inplace(1.0 / rho);
                            let ct_bar_shape = ct_bar.shape();

                            let mut u = empty_array();
                            u.r_mut()
                                .fill_from_resize(r.r().into_subview([0, rank - 1], [rank - 1, 1]));
                            println!("29");
                            // Modify R
                            permute_matrix(&mut r, rank - 1, rank, Some(rank - 1));
                            r.r_mut()[[rank - 1, rank - 1]] = ga_bar;
                            r.r_mut()[[rank - 1, rank]] = ga * mu / rho;
                            r.r_mut()[[rank, rank]] = ga * nu / rho;

                            if r_shape[1] - (rank + 1) > 0 {
                                r.r_mut()
                                    .into_subview(
                                        [rank - 1, rank + 1],
                                        [2, r_shape[1] - (rank + 1)],
                                    )
                                    .fill_from(ct_bar.r());
                                println!("30");
                            }

                            let r11_tmp = TriangularMatrix::<$scalar>::new(
                                &r.r().into_subview([0, 0], [rank - 1, rank - 1]),
                                TriangularType::Upper,
                            )
                            .unwrap();

                            let mut u1 = empty_array();
                            let r12_shape = r12.shape();

                            r11_tmp.solve(&mut u, Side::Left, TransMode::NoTrans);
                            u1.r_mut()
                                .fill_from_resize(r12.r().into_subview([0, 0], [rank - 1, 1]));

                            r12.r_mut().into_subview([0, 0], [rank - 1, 1]).fill_from(
                                ((nu * nu) / (rho * rho)) * u.r() - (mu / (rho * rho)) * u1.r(),
                            );
                            r12.r_mut()[[rank - 1, 0]] = mu / (rho * rho);
                            r12.r_mut()
                                .into_subview([rank - 1, 1], [1, r12_shape[1] - 1])
                                .fill_from(ct_bar.r().into_subview([0, 0], [1, ct_bar_shape[1]]));

                            if r_shape[1] - (rank + 1) > 0 {
                                r12.r_mut()
                                    .into_subview([rank - 1, 1], [1, r12_shape[1] - 1])
                                    .scale_inplace(1.0 / ga_bar);
                            }
                            println!("31, {:?}, {}, {}", ct_bar.shape(), 1, ct_bar_shape[1]);

                            let mut tmp1: DynamicArray<$scalar, 2> = empty_array();
                            let mut tmp2: DynamicArray<$scalar, 2> = empty_array();
                            let mut tmp3: DynamicArray<$scalar, 2> = empty_array();

                            if r_shape[1] - (rank + 1) > 0 {
                                tmp1.r_mut().mult_into_resize(
                                    TransMode::NoTrans,
                                    TransMode::NoTrans,
                                    nu / ga_bar,
                                    u.r(),
                                    ct_bar.r().into_subview([1, 0], [1, ct_bar_shape[1]]),
                                    num::Zero::zero(),
                                );

                                println!("31.0");
                                tmp2.r_mut().mult_into_resize(
                                    TransMode::NoTrans,
                                    TransMode::NoTrans,
                                    -1.0 / ga_bar,
                                    u1.r(),
                                    ct_bar.r().into_subview([0, 0], [1, ct_bar_shape[1]]),
                                    num::Zero::zero(),
                                );

                                println!(
                                    "31.1, {:?}, {}, {}",
                                    r12.r().shape(),
                                    rank - 1,
                                    r12_shape[1] - 1
                                );

                                tmp3.r_mut().fill_from_resize(
                                    r12.r().into_subview([0, 1], [rank - 1, r12_shape[1] - 1]),
                                );
                                r12.r_mut()
                                    .into_subview([0, 1], [rank - 1, r12_shape[1] - 1])
                                    .fill_from(tmp1.r() + tmp2.r() + tmp3.r());
                            }
                            println!("32");
                            gamma[0] = ga * nu / rho;

                            for i in 1..gamma.len() {
                                gamma[i] = ((gamma[i].powi(2)
                                    + ct_bar.r().slice(0, 1)[[i - 1]].powi(2)
                                    - ct.r().slice(0, 1)[[i - 1]].powi(2))
                                .max(0.0))
                                .sqrt();
                            }

                            let mut u_bar = empty_array();
                            u_bar.r_mut().fill_from_resize(u1.r() + mu * u.r());

                            omega[rank - 1] = ga_bar;

                            for i in 0..(rank - 1) {
                                let val = omega[i].powi(-2)
                                    + u_bar.r().data()[i].powi(2) / (ga_bar * ga_bar)
                                    - u.r().data()[i].powi(2) / (ga * ga);
                                omega[i] = 1.0 / val.sqrt();
                            }
                            println!("33");
                            let givens_rotation = GivensRotationData {
                                c: -mu / rho,
                                s: -nu / rho,
                                r: 1.0,
                            };

                            if rank < r_shape[0] {
                                let g = givens_rotation.get_givens_matrix();
                                let q_shape = q.shape();
                                let mut q_block =
                                    q.r_mut().into_subview([0, rank - 1], [q_shape[0], 2]);
                                let mut gq: DynamicArray<$scalar, 2> = empty_array();
                                gq.r_mut().mult_into_resize(
                                    TransMode::NoTrans,
                                    TransMode::ConjTrans,
                                    num::One::one(),
                                    q_block.r(),
                                    g.r(),
                                    num::Zero::zero(),
                                );
                                q_block.r_mut().fill_from(gq.r());
                            }
                            println!("34");
                        }

                        let (i, &min_tmp) = omega
                            .iter()
                            .enumerate()
                            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                            .unwrap();
                        println!("35");
                        match tol_type {
                            QrTolerance::Abs => {
                                if min_tmp > tol {
                                    break;
                                }
                            }
                            QrTolerance::Rel => {
                                let &max_tmp = omega
                                    .iter()
                                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                                    .unwrap();
                                if min_tmp / max_tmp > tol {
                                    break;
                                }
                            }
                        }
                        println!("36");
                        if i < rank - 1 {
                            let mut order: Vec<usize> = (i + 1..=rank - 1).collect();
                            order.push(i);

                            for (dest, src) in (i..=rank - 1).zip(order.iter().copied()) {
                                perm[dest] = perm[src];
                            }

                            for row in 0..r_shape[0] {
                                let reordered: Vec<$scalar> =
                                    order.iter().map(|&col| r[[row, col]]).collect();
                                for (dest, val) in (i..=rank - 1).zip(reordered) {
                                    r.r_mut()[[row, dest]] = val;
                                }
                            }
                            //Givens rotation for the triangulation of R(1:k, 1:k)
                            for ind in i..rank - 1 {
                                let mut r_block = r.r_mut().into_subview([ind, 0], [2, r_shape[1]]);
                                let vec_r = r_block.r().into_subview([0, ind], [2, 1]);
                                let mut givens_rotation: GivensRotationData<$scalar> =
                                    GivensRotation::<$scalar>::new(vec_r[[0, 0]], vec_r[[1, 0]]);
                                let tmp = givens_rotation.c * vec_r[[0, 0]]
                                    + givens_rotation.s * vec_r[[1, 0]];

                                if tmp < 0.0 {
                                    // guarantee r(ind, ind) > 0
                                    givens_rotation.c = -givens_rotation.c;
                                    givens_rotation.s = -givens_rotation.s;
                                    givens_rotation.r = -givens_rotation.r;
                                }

                                let g = givens_rotation.get_givens_matrix();
                                let mut rg: DynamicArray<$scalar, 2> = empty_array();
                                rg.r_mut().simple_mult_into_resize(g.r(), r_block.r());
                                r_block.r_mut().fill_from(rg.r());

                                let mut q_block = q.r_mut().into_subview([0, ind], [q_shape[0], 2]);
                                let mut gq: DynamicArray<$scalar, 2> = empty_array();
                                gq.r_mut().mult_into_resize(
                                    TransMode::NoTrans,
                                    TransMode::ConjTrans,
                                    num::One::one(),
                                    q_block.r(),
                                    g.r(),
                                    num::Zero::zero(),
                                );
                                q_block.r_mut().fill_from(gq.r());
                                if r.r()[[rank - 1, rank - 1]] < 0.0 {
                                    r.r_mut()
                                        .into_subview([rank - 1, 0], [1, r_shape[1]])
                                        .scale_inplace(-1.0);
                                    q.r_mut()
                                        .into_subview([0, rank - 1], [r_shape[0], 1])
                                        .scale_inplace(-1.0);
                                }
                            }
                        }
                        println!("37");
                        rank -= 1;

                        //Update A^{-1}B, omega, gamma
                        (r11, r12, r22) = get_r_blocks(&r, dim, rank);
                        r11_inv = rlst_dynamic_array2!($scalar, [rank, rank]);
                        r11_inv.r_mut().set_identity();
                        r11.solve(&mut r11_inv, Side::Left, TransMode::NoTrans);
                        r11.solve(&mut r12, Side::Left, TransMode::NoTrans);

                        if rank == 1 {
                            break;
                        }

                        gamma = (0..dim - rank)
                            .map(|j| r22.r().slice(1, j).norm_2())
                            .collect::<Vec<_>>();

                        omega = (0..rank)
                            .map(|i| 1.0 / r11_inv.r().slice(0, i).norm_2())
                            .collect::<Vec<_>>();
                        println!("38");
                    }
                    println!("39");
                    let mut q_trunc = empty_array();
                    let mut r_trunc = empty_array();
                    q_trunc
                        .r_mut()
                        .fill_from_resize(q.r().into_subview([0, 0], [q_shape[0], rank]));
                    r_trunc
                        .r_mut()
                        .fill_from_resize(r.r().into_subview([0, 0], [rank, r_shape[1]]));
                    println!("40");
                    (q_trunc, r_trunc, perm, rank)
                } else {
                    (q, r, perm, rank)
                }
            }
        }
    };
}

macro_rules! implement_special_qr_complex {
    ($scalar:ty, $geqp3:expr, $geqrf: expr, $ormqr:expr) => {
        impl RankRevealingMatrixQrDecomposition for RankRevealingQrDecomposition<$scalar> {
            type Item = $scalar;

            fn new<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                arr: Array<$scalar, ArrayImplMut, 2>,
                qr_type: RankRevealingQrType<<$scalar as RlstScalar>::Real>,
                rank_param: RankParam<<$scalar as RlstScalar>::Real>,
            ) -> Self {
                let (q, r, perm, rank) = match qr_type {
                    RankRevealingQrType::RRQR => Self::rrqr(arr, rank_param),
                    RankRevealingQrType::SRRQR(f) => Self::srrqr(arr, rank_param, f),
                };

                Self { q, r, perm, rank }
            }

            fn rrqr<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                mut arr: Array<$scalar, ArrayImplMut, 2>,
                rank_param: RankParam<<$scalar as RlstScalar>::Real>,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                let [m, n] = arr.shape();
                let qr = arr.r_mut().into_qr_alloc(Pivoting::True).unwrap();
                let perm = qr.get_perm();

                let (mut r, mut q) = if m < n {
                    let mut r: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [m, n]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, m]);
                    qr.get_r(r.r_mut());
                    let _ = qr.get_q_alloc(q.r_mut());
                    (r, q)
                } else {
                    let mut r: DynamicArray<$scalar, 2> = rlst_dynamic_array2!($scalar, [n, n]);
                    let mut q = rlst_dynamic_array2!($scalar, [m, n]);
                    qr.get_r(r.r_mut());
                    let _ = qr.get_q_alloc(q.r_mut());
                    (r, q)
                };

                let rank = match rank_param {
                    RankParam::Tol(tol, tol_type) => match tol_type {
                        QrTolerance::Abs => rank_from_tolerance_abs(&mut r, &mut q, tol),
                        QrTolerance::Rel => rank_from_tolerance_rel(&mut r, &mut q, tol),
                    },
                    RankParam::Rank(rank) => rank,
                };

                (q, r, perm, rank)
            }

            fn srrqr<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                _arr: Array<$scalar, ArrayImplMut, 2>,
                _rank_param: RankParam<<$scalar as RlstScalar>::Real>,
                _f: <$scalar as RlstScalar>::Real,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                panic!("SRRQR not implemented for complex matrices");
            }

            fn srrqr_k<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                _arr: Array<$scalar, ArrayImplMut, 2>,
                _f: <$scalar as RlstScalar>::Real,
                _rank: usize,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                panic!("SRRQR not implemented for complex matrices");
            }

            fn srrqr_tol<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            >(
                _arr: Array<$scalar, ArrayImplMut, 2>,
                _f: <$scalar as RlstScalar>::Real,
                _tol: <$scalar as RlstScalar>::Real,
                _tol_type: QrTolerance,
            ) -> (
                DynamicArray<$scalar, 2>,
                DynamicArray<$scalar, 2>,
                Vec<usize>,
                usize,
            ) {
                panic!("SRRQR not implemented for complex matrices");
            }
        }
    };
}

implement_special_qr_real!(f64, dgeqp3, dgeqrf, dormqr);
implement_special_qr_real!(f32, sgeqp3, sgeqrf, sormqr);
implement_special_qr_complex!(c64, zgeqp3, zgeqrf, zunmqr);
implement_special_qr_complex!(c32, cgeqp3, cgeqrf, cunmqr);

fn sign<Item: RlstScalar>(z: Item) -> Item {
    if z.abs() == num::Zero::zero() {
        z.clone()
    } else {
        z.clone() / Item::from(z.abs()).unwrap()
    }
}

fn rank_from_tolerance_abs<Item: RlstScalar>(
    r: &mut DynamicArray<Item, 2>,
    q: &mut DynamicArray<Item, 2>,
    tol: <Item as RlstScalar>::Real,
) -> usize {
    let [rows, cols] = r.shape();

    // Step 1: ss = sign of diagonal
    let ss: Vec<Item> = if rows == 1 || cols == 1 {
        vec![sign(r.r()[[0, 0]].clone())]
    } else {
        (0..rows.min(cols))
            .map(|i| sign(r.r()[[i, i]].clone()))
            .collect()
    };

    // Step 2: scale R rows
    for i in 0..rows {
        let factor = ss[i].clone();
        for j in 0..cols {
            r.r_mut()[[i, j]] *= factor.clone();
        }
    }

    // Step 3: scale Q columns
    let [q_rows, q_cols] = q.shape();
    for j in 0..q_cols {
        let factor = ss[j].clone();
        for i in 0..q_rows {
            q.r_mut()[[i, j]] *= factor.clone();
        }
    }

    // Step 4: find k
    let mut k: usize = 0;
    if rows == 1 || cols == 1 {
        if r.r()[[0, 0]].abs() > tol {
            k = 1;
        }
    } else {
        for i in (0..rows.min(cols)).rev() {
            if r.r()[[i, i]].abs() > tol {
                k = i + 1;
                break;
            }
        }
    }
    k
}

fn rank_from_tolerance_rel<Item: RlstScalar>(
    r: &mut DynamicArray<Item, 2>,
    q: &mut DynamicArray<Item, 2>,
    tol: <Item as RlstScalar>::Real,
) -> usize {
    let [rows, cols] = r.shape();

    // Step 1: compute ss
    let ss: Vec<Item> = if rows == 1 || cols == 1 {
        vec![sign(r.r()[[0, 0]].clone())]
    } else {
        (0..rows.min(cols))
            .map(|i| sign(r.r()[[i, i]].clone()))
            .collect()
    };

    // Step 2: scale R rows
    for i in 0..rows {
        let factor = ss[i].clone();
        for j in 0..cols {
            r.r_mut()[[i, j]] *= factor.clone();
        }
    }

    // Step 3: scale Q columns
    let [q_rows, q_cols] = q.shape();
    for j in 0..q_cols {
        let factor = ss[j].clone();
        for i in 0..q_rows {
            q.r_mut()[[i, j]] *= factor.clone();
        }
    }

    // Step 4: find k using relative tolerance
    let mut k: usize = 0;
    let ref_val = r.r()[[0, 0]].abs(); // baseline for relative check
    if ref_val == num::Zero::zero() {
        return 0;
    }

    if rows == 1 || cols == 1 {
        if r.r()[[0, 0]].abs() / ref_val > tol {
            k = 1;
        }
    } else {
        for i in (0..rows.min(cols)).rev() {
            if r.r()[[i, i]].abs() / ref_val > tol {
                k = i + 1;
                break;
            }
        }
    }
    k
}
