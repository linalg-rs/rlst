//! LU Decomposition and linear system solves.
use super::assert_lapack_stride;
use crate::dense::array::Array;
use crate::dense::traits::{
    RandomAccessByValue, RawAccess, RawAccessMut, ResizeInPlace, Shape, Stride,
    UnsafeRandom1DAccessMut, UnsafeRandomAccessByRef, UnsafeRandomAccessByValue,
    UnsafeRandomAccessMut,
};
use crate::dense::types::{c32, c64, RlstError, RlstResult, RlstScalar, TransMode};
use lapack::{cgetrf, cgetrs, dgetrf, dgetrs, sgetrf, sgetrs, zgetrf, zgetrs};
use num::One;

/// Compute an LU decomposition from a given two-dimensional array.
pub trait MatrixLu: RlstScalar {
    /// Compute the matrix inverse
    fn into_lu_alloc<
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
            + Stride<2>
            + Shape<2>
            + RawAccessMut<Item = Self>,
    >(
        arr: Array<Self, ArrayImpl, 2>,
    ) -> RlstResult<LuDecomposition<Self, ArrayImpl>>;
}

macro_rules! implement_into_lu {
    ($scalar:ty) => {
        impl MatrixLu for $scalar {
            fn into_lu_alloc<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = Self>,
            >(
                arr: Array<Self, ArrayImpl, 2>,
            ) -> RlstResult<LuDecomposition<Self, ArrayImpl>> {
                LuDecomposition::<$scalar, ArrayImpl>::new(arr)
            }
        }
    };
}

implement_into_lu!(f32);
implement_into_lu!(f64);
implement_into_lu!(c32);
implement_into_lu!(c64);

impl<
        Item: RlstScalar + MatrixLu,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item>
            + Stride<2>
            + RawAccessMut<Item = Item>
            + Shape<2>,
    > Array<Item, ArrayImpl, 2>
{
    /// Compute the LU decomposition of a matrix.
    ///
    /// The LU Decomposition of an `(m,n)` matrix `A` is defined
    /// by `A = PLU`, where `P` is an `(m, m)` permutation matrix,
    /// `L` is a `(m, k)` unit lower triangular matrix, and `U` is
    /// an `(k, n)` upper triangular matrix.
    pub fn into_lu_alloc(self) -> RlstResult<LuDecomposition<Item, ArrayImpl>> {
        <Item as MatrixLu>::into_lu_alloc(self)
    }
}

/// Compute the LU decomposition of a matrix.
///
/// The LU Decomposition of an `(m,n)` matrix `A` is defined
/// by `A = PLU`, where `P` is an `(m, m)` permutation matrix,
/// `L` is a `(m, k)` unit lower triangular matrix, and `U` is
/// an `(k, n)` upper triangular matrix.
pub trait MatrixLuDecomposition: Sized {
    /// Item type
    type Item: RlstScalar;
    /// Array implementaion
    type ArrayImpl: UnsafeRandomAccessByValue<2, Item = Self::Item>
        + Stride<2>
        + RawAccessMut<Item = Self::Item>
        + Shape<2>;

    /// Create a new LU Decomposition from a given array.
    fn new(arr: Array<Self::Item, Self::ArrayImpl, 2>) -> RlstResult<Self>;

    /// Solve a linear system with a single right-hand side.
    ///
    /// The right-hand side is overwritten with the solution.
    fn solve_vec<
        ArrayImplMut: RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessByValue<1, Item = Self::Item>
            + Shape<1>
            + Stride<1>,
    >(
        &self,
        trans: TransMode,
        rhs: Array<Self::Item, ArrayImplMut, 1>,
    ) -> RlstResult<()>;

    /// Solve a linear system with multiple right-hand sides.
    ///
    /// The right-hand sides are overwritten with the solution.
    fn solve_mat<
        ArrayImplMut: RawAccessMut<Item = Self::Item>
            + UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + Stride<2>,
    >(
        &self,
        trans: TransMode,
        rhs: Array<Self::Item, ArrayImplMut, 2>,
    ) -> RlstResult<()>;

    /// Get the L matrix of the LU Decomposition.
    ///
    /// This method resizes the input `arr` as required.
    fn get_l_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + ResizeInPlace<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Get the L matrix of the LU decomposition.
    ///
    /// If A has the dimension `(m, n)` then the L matrix
    /// has the dimension `(m, k)` with `k = min(m, n)`.
    fn get_l<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Get the R matrix of the LU Decomposition.
    ///
    /// This method resizes the input `arr` as required.
    fn get_u_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + ResizeInPlace<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Get the R matrix of the LU Decomposition.
    ///
    /// If A has the dimension `(m, n)` then the L matrix
    /// has the dimension `(k, n)` with `k = min(m, n)`.
    fn get_u<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Get the P matrix of the LU Decomposition.
    ///
    /// This method resizes the input `arr` as required.
    fn get_p_resize<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>
            + ResizeInPlace<2>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Get the permutation vector from the LU decomposition.
    ///
    /// If `perm[i] = j` then the ith row of `LU` corresponds to the jth row of `A`.
    fn get_perm(&self) -> Vec<usize>;

    /// Get the P matrix of the LU Decomposition.
    ///
    /// If A has the dimension `(m, n)` then the P matrix
    /// has the dimension (m, m).
    fn get_p<
        ArrayImplMut: UnsafeRandomAccessByValue<2, Item = Self::Item>
            + Shape<2>
            + UnsafeRandomAccessMut<2, Item = Self::Item>
            + UnsafeRandomAccessByRef<2, Item = Self::Item>,
    >(
        &self,
        arr: Array<Self::Item, ArrayImplMut, 2>,
    );

    /// Compute the determinant of A.
    fn det(&self) -> Self::Item;
}

/// Container for the LU Decomposition of a matrix.
pub struct LuDecomposition<
    Item: RlstScalar,
    ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Stride<2> + Shape<2> + RawAccessMut<Item = Item>,
> {
    arr: Array<Item, ArrayImpl, 2>,
    ipiv: Vec<i32>,
}

macro_rules! impl_lu {
    ($scalar:ty, $getrf:expr, $getrs:expr) => {
        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + Shape<2>
                    + RawAccessMut<Item = $scalar>,
            > MatrixLuDecomposition for LuDecomposition<$scalar, ArrayImpl>
        {
            type Item = $scalar;
            type ArrayImpl = ArrayImpl;

            fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
                let shape = arr.shape();
                let stride = arr.stride();

                assert_lapack_stride(stride);

                let dim = std::cmp::min(shape[0], shape[1]);
                if dim == 0 {
                    return Err(RlstError::MatrixIsEmpty((shape[0], shape[1])));
                }
                let mut ipiv = vec![0; dim];
                let mut info = 0;
                unsafe {
                    $getrf(
                        shape[0] as i32,
                        shape[1] as i32,
                        arr.data_mut(),
                        stride[1] as i32,
                        &mut ipiv,
                        &mut info,
                    );
                }

                match info {
                    0 => Ok(Self { arr, ipiv }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }

            fn solve_vec<
                ArrayImplMut: RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessByValue<1, Item = $scalar>
                    + Shape<1>
                    + Stride<1>,
            >(
                &self,
                trans: TransMode,
                rhs: Array<$scalar, ArrayImplMut, 1>,
            ) -> RlstResult<()> {
                self.solve_mat(
                    trans,
                    rhs.insert_empty_axis(crate::dense::array::empty_axis::AxisPosition::Back),
                )
            }

            fn solve_mat<
                ArrayImplMut: RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                trans: TransMode,
                mut rhs: Array<$scalar, ArrayImplMut, 2>,
            ) -> RlstResult<()> {
                assert_eq!(self.arr.shape()[0], self.arr.shape()[1]);
                let n = self.arr.shape()[0];
                assert_eq!(rhs.shape()[0], n);

                let nrhs = rhs.shape()[1];

                let arr_stride = self.arr.stride();
                let rhs_stride = rhs.stride();

                let lda = self.arr.stride()[1];
                let ldb = rhs.stride()[1];

                assert_lapack_stride(arr_stride);
                assert_lapack_stride(rhs_stride);

                let trans_param = match trans {
                    TransMode::NoTrans => b'N',
                    TransMode::Trans => b'T',
                    TransMode::ConjTrans => b'C',
                    _ => panic!("Transposition mode not supported for LU solve."),
                };

                let mut info = 0;
                unsafe {
                    $getrs(
                        trans_param,
                        n as i32,
                        nrhs as i32,
                        self.arr.data(),
                        lda as i32,
                        self.ipiv.as_slice(),
                        rhs.data_mut(),
                        ldb as i32,
                        &mut info,
                    )
                };

                match info {
                    0 => Ok(()),
                    _ => Err(RlstError::LapackError(info)),
                }
            }

            fn get_l_resize<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];
                let n = self.arr.shape()[1];
                let k = std::cmp::min(m, n);

                arr.resize_in_place([m, k]);
                self.get_l(arr);
            }

            fn get_l<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];
                let n = self.arr.shape()[1];
                let k = std::cmp::min(m, n);
                assert_eq!(
                    arr.shape(),
                    [m, k],
                    "Require matrix with shape {} x {}. Given shape is {} x {}",
                    m,
                    k,
                    arr.shape()[0],
                    arr.shape()[1]
                );

                arr.set_zero();
                for col in 0..k {
                    for row in col..m {
                        if col == row {
                            arr[[row, col]] = <$scalar as One>::one();
                        } else {
                            arr[[row, col]] = self.arr.get_value([row, col]).unwrap();
                        }
                    }
                }
            }

            fn get_u_resize<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];
                let n = self.arr.shape()[1];
                let k = std::cmp::min(m, n);

                arr.resize_in_place([k, n]);
                self.get_u(arr);
            }

            fn get_u<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];
                let n = self.arr.shape()[1];
                let k = std::cmp::min(m, n);
                assert_eq!(
                    arr.shape(),
                    [k, n],
                    "Require matrix with shape {} x {}. Given shape is {} x {}",
                    k,
                    n,
                    arr.shape()[0],
                    arr.shape()[1]
                );

                arr.set_zero();
                for col in 0..n {
                    for row in 0..=std::cmp::min(col, k - 1) {
                        arr[[row, col]] = self.arr.get_value([row, col]).unwrap();
                    }
                }
            }

            fn get_p_resize<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>
                    + ResizeInPlace<2>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];

                arr.resize_in_place([m, m]);
                self.get_p(arr);
            }

            /// Get the permutation vector from the LU decomposition.
            ///
            /// If `perm[i] = j` then the ith row of `LU` corresponds to the jth row of `A`.
            fn get_perm(&self) -> Vec<usize> {
                let m = self.arr.shape()[0];
                let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

                let mut perm = (0..m).collect::<Vec<_>>();

                for (index, &elem) in ipiv.iter().enumerate() {
                    perm.swap(index, elem);
                }

                perm
            }

            fn get_p<
                ArrayImplMut: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + UnsafeRandomAccessMut<2, Item = $scalar>
                    + UnsafeRandomAccessByRef<2, Item = $scalar>,
            >(
                &self,
                mut arr: Array<$scalar, ArrayImplMut, 2>,
            ) {
                let m = self.arr.shape()[0];
                assert_eq!(
                    arr.shape(),
                    [m, m],
                    "Require matrix with shape {} x {}. Given shape is {} x {}",
                    m,
                    m,
                    arr.shape()[0],
                    arr.shape()[1]
                );

                let perm = self.get_perm();

                arr.set_zero();
                for col in 0..m {
                    arr[[perm[col], col]] = <$scalar as One>::one();
                }
            }

            fn det(&self) -> Self::Item {
                assert_eq!(self.arr.shape()[0], self.arr.shape()[1]);
                let n = self.arr.shape()[0];

                let mut det = <$scalar as One>::one();
                for index in 0..n {
                    det *= self.arr.get_value([index, index]).unwrap();
                    // Every permutation changes the sign of the determinant.
                    // Need to compare with 1 + index because ipiv is in Fortran numbering.
                    if (1 + index) as i32 != self.ipiv[index] {
                        det = -det;
                    }
                }

                det
            }
        }
    };
}

impl_lu!(f64, dgetrf, dgetrs);
impl_lu!(f32, sgetrf, sgetrs);
impl_lu!(c64, zgetrf, zgetrs);
impl_lu!(c32, cgetrf, cgetrs);
