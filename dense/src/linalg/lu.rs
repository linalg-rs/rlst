//! LU Decomposition and linear system solves.
use super::assert_lapack_stride;
use crate::array::Array;
use crate::traits::*;
use lapack::{cgetrf, cgetrs, dgetrf, dgetrs, sgetrf, sgetrs, zgetrf, zgetrs};
use num::One;
use rlst_common::types::*;

/// Transposition modes for solving linear systems via LU decomposition.
pub enum LuTrans {
    /// Transpose.
    Trans,
    /// No transpose.
    NoTrans,
    /// Conjugate transpose.
    ConjTrans,
}

/// Container for the LU Decomposition of a matrix.
pub struct LuDecomposition<
    Item: Scalar,
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
            > LuDecomposition<$scalar, ArrayImpl>
        {
            /// Create a new LU Decomposition from a given array.
            pub fn new(mut arr: Array<$scalar, ArrayImpl, 2>) -> RlstResult<Self> {
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

            /// Solve a linear system with a single right-hand side.
            ///
            /// The right-hand side is overwritten with the solution.
            pub fn solve_vec<
                ArrayImplMut: RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessByValue<1, Item = $scalar>
                    + Shape<1>
                    + Stride<1>,
            >(
                &self,
                trans: LuTrans,
                rhs: Array<$scalar, ArrayImplMut, 1>,
            ) -> RlstResult<()> {
                self.solve_mat(
                    trans,
                    rhs.insert_empty_axis(crate::array::empty_axis::AxisPosition::Back),
                )
            }

            /// Solve a linear system with multiple right-hand sides.
            ///
            /// The right-hand sides are overwritten with the solution.
            pub fn solve_mat<
                ArrayImplMut: RawAccessMut<Item = $scalar>
                    + UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Shape<2>
                    + Stride<2>,
            >(
                &self,
                trans: LuTrans,
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
                    LuTrans::NoTrans => b'N',
                    LuTrans::Trans => b'T',
                    LuTrans::ConjTrans => b'C',
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

            /// Get the L matrix of the LU Decomposition.
            ///
            /// This method resizes the input `arr` as required.
            pub fn get_l_resize<
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

            /// Get the L matrix of the LU decomposition.
            ///
            /// If A has the dimension `(m, n)` then the L matrix
            /// has the dimension `(m, k)` with `k = min(m, n)`.
            pub fn get_l<
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

            /// Get the R matrix of the LU Decomposition.
            ///
            /// This method resizes the input `arr` as required.
            pub fn get_u_resize<
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

            /// Get the R matrix of the LU Decomposition.
            ///
            /// If A has the dimension `(m, n)` then the L matrix
            /// has the dimension `(k, n)` with `k = min(m, n)`.
            pub fn get_u<
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

            /// Get the P matrix of the LU Decomposition.
            ///
            /// This method resizes the input `arr` as required.
            pub fn get_p_resize<
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

            /// Get the P matrix of the LU Decomposition.
            ///
            /// If A has the dimension `(m, n)` then the P matrix
            /// has the dimension (m, m).
            pub fn get_p<
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
        }

        impl<
                ArrayImpl: UnsafeRandomAccessByValue<2, Item = $scalar>
                    + Stride<2>
                    + RawAccessMut<Item = $scalar>
                    + Shape<2>,
            > Array<$scalar, ArrayImpl, 2>
        {
            /// Compute the LU decomposition of a matrix.
            ///
            /// The LU Decomposition of an `(m,n)` matrix `A` is defined
            /// by `A = PLU`, where `P` is an `(m, m)` permutation matrix,
            /// `L` is a `(m, k)` unit lower triangular matrix, and `U` is
            /// an `(k, n)` upper triangular matrix.
            pub fn into_lu(self) -> RlstResult<LuDecomposition<$scalar, ArrayImpl>> {
                assert!(!self.is_empty(), "Matrix is empty.");
                LuDecomposition::<$scalar, ArrayImpl>::new(self)
            }
        }
    };
}

impl_lu!(f64, dgetrf, dgetrs);
impl_lu!(f32, sgetrf, sgetrs);
impl_lu!(c64, zgetrf, zgetrs);
impl_lu!(c32, cgetrf, cgetrs);

#[cfg(test)]
mod test {

    use crate::assert_array_relative_eq;

    use crate::rlst_dynamic_array1;
    use crate::rlst_dynamic_array2;

    use super::*;
    use crate::array::empty_array;

    use paste::paste;

    macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let dim = [8, 20];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = arr2.into_lu().unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = crate::array::empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_square_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = arr2.into_lu().unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = crate::array::empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_solve_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);
                    arr.fill_from_seed_equally_distributed(0);
                    let mut x_actual = rlst_dynamic_array1!($scalar, [dim[0]]);
                    let mut rhs = rlst_dynamic_array1!($scalar, [dim[0]]);
                    x_actual.fill_from_seed_equally_distributed(1);
                    rhs.view_mut().simple_mult_into_resize(arr.view(), x_actual.view());

                    let lu = arr.into_lu().unwrap();
                    lu.solve_vec(LuTrans::NoTrans, rhs.view_mut()).unwrap();

                    assert_array_relative_eq!(x_actual, rhs, $tol)
                }



                #[test]
                fn [<test_lu_thin_$scalar>]() {
                    let dim = [12, 8];
                    let mut arr = rlst_dynamic_array2!($scalar, dim);

                    arr.fill_from_seed_normally_distributed(0);
                    let mut arr2 = rlst_dynamic_array2!($scalar, dim);
                    arr2.fill_from(arr.view());

                    let lu = arr2.into_lu().unwrap();

                    let mut l_mat = empty_array::<$scalar, 2>();
                    let mut u_mat = empty_array::<$scalar, 2>();
                    let mut p_mat = empty_array::<$scalar, 2>();

                    lu.get_l_resize(l_mat.view_mut());
                    lu.get_u_resize(u_mat.view_mut());
                    lu.get_p_resize(p_mat.view_mut());

                    let res = crate::array::empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    assert_array_relative_eq!(res, arr, $tol)
                }

            }
        };
    }

    impl_lu_tests!(f64, 1E-12);
    impl_lu_tests!(f32, 1E-5);
    impl_lu_tests!(c64, 1E-12);
    impl_lu_tests!(c32, 1E-5);
}
