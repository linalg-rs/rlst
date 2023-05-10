//! Interface to Lapack LU Decomposition.
//!
//! The Lapack LU interface uses the Lapack routines `_getrf` to compute an LU decomposition and
//! `_getrs` to solve a linear system of equations.

use crate::lapack::LapackData;
use crate::traits::lu_decomp::{LUDecomp, LU};
use lapacke;
use num::traits::One;
use rlst_common::traits::Copy;
use rlst_common::types::{c32, c64, RlstError, RlstResult, Scalar};
use rlst_dense::{rlst_mat, traits::*, MatrixD};

use super::{check_lapack_stride, TransposeMode};
use crate::linalg::DenseMatrixLinAlgBuilder;
use std::marker::PhantomData;

pub struct LUDecompLapack<T: Scalar, Mat: RawAccessMut<T = T> + Shape + Stride> {
    data: LapackData<<Mat as RawAccess>::T, Mat>,
    ipiv: Vec<i32>,
    _marker: PhantomData<T>,
}

macro_rules! lu_decomp_impl {
    ($scalar:ty, $lapack_getrf:ident, $lapack_getrs:ident) => {
        impl<'a, Mat: Copy> LU for DenseMatrixLinAlgBuilder<'a, $scalar, Mat>
        where
            <Mat as Copy>::Out: RawAccessMut<T = $scalar> + Shape + Stride,
        {
            type T = $scalar;
            type Out = LUDecompLapack<$scalar, <Mat as Copy>::Out>;
            /// Compute the LU decomposition.
            fn lu(self) -> RlstResult<LUDecompLapack<$scalar, <Mat as Copy>::Out>> {
                let mut copied = self.into_lapack()?;
                let shape = copied.mat.shape();
                let stride = copied.mat.stride();

                let m = shape.0 as i32;
                let n = shape.1 as i32;
                let lda = stride.1 as i32;

                let mut ipiv: Vec<i32> = vec![0; std::cmp::min(shape.0, shape.1)];
                let info = unsafe {
                    lapacke::$lapack_getrf(
                        lapacke::Layout::ColumnMajor,
                        m,
                        n,
                        copied.mat.data_mut(),
                        lda,
                        ipiv.as_mut_slice(),
                    )
                };
                if info == 0 {
                    return Ok(LUDecompLapack {
                        data: copied,
                        ipiv,
                        _marker: PhantomData,
                    });
                } else {
                    return Err(RlstError::LapackError(info));
                }
            }
        }

        impl<
                Mat: RawAccessMut<T = $scalar>
                    + Shape
                    + Stride
                    + std::ops::Index<[usize; 2], Output = $scalar>
                    + std::ops::IndexMut<[usize; 2], Output = $scalar>
                    + RawAccess<T = $scalar>,
            > LUDecomp for LUDecompLapack<$scalar, Mat>
        {
            type T = $scalar;

            fn data(&self) -> &[Self::T] {
                self.data.mat.data()
            }

            fn shape(&self) -> (usize, usize) {
                self.data.mat.shape()
            }

            fn get_l(&self) -> MatrixD<Self::T> {
                let shape = self.shape();
                let dim = std::cmp::min(shape.0, shape.1);
                let mut mat = rlst_mat!(Self::T, (shape.0, dim));

                for col in 0..dim {
                    for row in (1 + col)..shape.0 {
                        mat[[row, col]] = self.data.mat[[row, col]];
                    }
                }

                for index in 0..dim {
                    mat[[index, index]] = <Self::T as One>::one();
                }
                mat
            }

            fn get_perm(&self) -> Vec<usize> {
                let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();
                let mut perm = (0..self.shape().0).collect::<Vec<_>>();

                for index in 0..ipiv.len() {
                    let t = perm[index];
                    perm[index] = perm[ipiv[index]];
                    perm[ipiv[index]] = t;
                }

                perm
            }

            fn get_u(&self) -> MatrixD<Self::T> {
                let shape = self.shape();
                let dim = std::cmp::min(shape.0, shape.1);
                let mut mat = rlst_mat!(Self::T, (dim, shape.1));

                for row in 0..dim {
                    for col in row..shape.1 {
                        mat[[row, col]] = self.data.mat[[row, col]];
                    }
                }
                mat
            }

            fn solve<Rhs: RawAccessMut<T = $scalar> + Shape + Stride>(
                &self,
                rhs: &mut Rhs,
                trans: TransposeMode,
            ) -> RlstResult<()> {
                if !check_lapack_stride(rhs.shape(), rhs.stride()) {
                    return Err(RlstError::IncompatibleStride);
                } else {
                    let mat = &self.data.mat;

                    if mat.shape().0 != rhs.shape().0 {
                        return Err(RlstError::SingleDimensionError {
                            expected: mat.shape().0,
                            actual: rhs.shape().0,
                        });
                    }

                    if rhs.shape().1 == 0 {
                        return Err(RlstError::MatrixIsEmpty(rhs.shape()));
                    }

                    let ldb = rhs.stride().1;

                    let info = unsafe {
                        lapacke::$lapack_getrs(
                            lapacke::Layout::ColumnMajor,
                            trans as u8,
                            mat.shape().1 as i32,
                            rhs.shape().1 as i32,
                            mat.data(),
                            mat.stride().1 as i32,
                            &self.ipiv,
                            rhs.data_mut(),
                            ldb as i32,
                        )
                    };

                    if info != 0 {
                        return Err(RlstError::LapackError(info));
                    } else {
                        Ok(())
                    }
                }
            }
        }
    };
}

lu_decomp_impl!(f64, dgetrf, dgetrs);
lu_decomp_impl!(f32, sgetrf, sgetrs);
lu_decomp_impl!(c32, cgetrf, cgetrs);
lu_decomp_impl!(c64, zgetrf, zgetrs);

#[cfg(test)]
mod test {
    use approx::assert_relative_eq;
    use approx::assert_ulps_eq;
    use rand::SeedableRng;

    use crate::linalg::LinAlg;

    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rlst_dense::Dot;

    #[test]
    fn test_lu_solve_f64() {
        let mut rlst_mat = rlst_dense::rlst_mat![f64, (2, 2)];
        let mut rlst_vec = rlst_dense::rlst_col_vec![f64, 2];

        rlst_mat[[0, 0]] = 1.0;
        rlst_mat[[0, 1]] = 1.0;
        rlst_mat[[1, 0]] = 3.0;
        rlst_mat[[1, 1]] = 1.0;

        rlst_vec[[0, 0]] = 2.3;
        rlst_vec[[1, 0]] = 7.1;

        let mut rhs = rlst_mat.dot(&rlst_vec);

        let _ = rlst_mat
            .linalg()
            .lu()
            .unwrap()
            .solve(&mut rhs, TransposeMode::NoTrans);

        let x = rhs;

        assert_ulps_eq![x[[0, 0]], 2.3, max_ulps = 10];
        assert_ulps_eq![x[[1, 0]], 7.1, max_ulps = 10];
    }

    #[test]
    fn test_thin_lu_decomp() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut mat = rlst_mat!(f64, (10, 6));

        mat.fill_from_standard_normal(&mut rng);

        let mat2 = mat.copy();

        let lu_decomp = mat2.linalg().lu().unwrap();

        let l_mat = lu_decomp.get_l();
        let u_mat = lu_decomp.get_u();
        let perm = lu_decomp.get_perm();

        let res = l_mat.dot(&u_mat);

        for row in 0..mat.shape().0 {
            for col in 0..mat.shape().1 {
                assert_relative_eq!(mat[[perm[row], col]], res[[row, col]], epsilon = 1E-14);
            }
        }
    }

    #[test]
    fn test_thick_lu_decomp() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);

        let mut mat = rlst_mat!(f64, (6, 10));

        mat.fill_from_standard_normal(&mut rng);

        let mat2 = mat.copy();

        let lu_decomp = mat2.linalg().lu().unwrap();

        let l_mat = lu_decomp.get_l();
        let u_mat = lu_decomp.get_u();
        let perm = lu_decomp.get_perm();

        let res = l_mat.dot(&u_mat);

        for row in 0..mat.shape().0 {
            for col in 0..mat.shape().1 {
                assert_relative_eq!(mat[[perm[row], col]], res[[row, col]], epsilon = 1E-14);
            }
        }
    }
}
