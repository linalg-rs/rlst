//! Lapack LU Decomposition.

use lapack::{cgetrf, cgetrs, dgetrf, dgetrs, sgetrf, sgetrs, zgetrf, zgetrs};

use super::{lapack_dims, LapackWrapper};

use crate::dense::array::DynArray;
use crate::dense::types::{c32, c64, TransMode};

use crate::dense::types::{RlstError, RlstResult};
use crate::{Array, RawAccess, RawAccessMut, Shape, Stride};

use crate::dense::traits::UnsafeRandomAccessMut;

use num::One;

/// A trait for computing the LU decomposition of a matrix in place.
pub trait LapackLu
where
    LuDecomposition<Self::Item, Self::ArrayImpl>:
        ComputedLu<Item = Self::Item, ArrayImpl = Self::ArrayImpl>,
{
    /// The item type.
    type Item;

    /// The item type contained in the matrix.
    type ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Self::Item>;

    /// Compute the LU Decomposition of the matrix.
    fn lu(self) -> RlstResult<LuDecomposition<Self::Item, Self::ArrayImpl>>;
}

/// Trait for functions on a computed LU decomposition.
pub trait ComputedLu {
    /// The item type of the LU decomposition.
    type Item;

    /// Array Implementation type.
    type ArrayImpl: Shape<2> + Stride<2> + RawAccess<Item = Self::Item>;

    /// Return the LU decomposition data.
    fn lu_data(&self) -> &LapackWrapper<Self::Item, Self::ArrayImpl>;

    /// Return the pivot indices. Indices are 1-based.
    fn ipiv(&self) -> &[i32];

    /// Solve for a given right-hand side vector.
    fn solve_vec<OtherArrayImpl>(
        &self,
        trans: TransMode,
        rhs: &mut Array<OtherArrayImpl, 1>,
    ) -> RlstResult<()>
    where
        OtherArrayImpl: Shape<1> + Stride<1> + RawAccessMut<Item = Self::Item>;

    /// Solve for a given right-hand side matrix.
    fn solve_mat<OtherArrayImpl>(
        &self,
        trans: TransMode,
        rhs: &mut Array<OtherArrayImpl, 2>,
    ) -> RlstResult<()>
    where
        OtherArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Self::Item>;

    /// Return the L matrix of the LU decomposition.
    fn get_l(&self) -> DynArray<Self::Item, 2>;

    /// Return the U matrix of the LU decomposition.
    fn get_u(&self) -> DynArray<Self::Item, 2>;

    /// Return the P matrix of the LU decomposition.
    fn get_p(&self) -> DynArray<Self::Item, 2>;

    /// Get the permutation vector from the LU decomposition.
    ///
    /// If `perm[i] = j` then the ith row of `LU` corresponds to the jth row of `A`.
    fn get_perm(&self) -> Vec<usize>;

    /// Compute the determinant of the matrix.
    fn det(&self) -> Self::Item;
}

/// Store the result of an LU decomposition of a matrix.
pub struct LuDecomposition<Item, ArrayImpl>
where
    ArrayImpl: Shape<2> + Stride<2> + RawAccess<Item = Item>,
{
    lu: LapackWrapper<Item, ArrayImpl>,
    ipiv: Vec<i32>,
}

macro_rules! implement_lu {
    ($scalar:ty, $getrf:expr, $getrs:expr) => {
        impl<ArrayImpl> LapackLu for LapackWrapper<$scalar, ArrayImpl>
        where
            ArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = $scalar>,
        {
            type Item = $scalar;

            type ArrayImpl = ArrayImpl;

            fn lu(mut self) -> RlstResult<LuDecomposition<$scalar, ArrayImpl>> {
                let (m, n, lda) = self.lapack_dims();

                let dim = std::cmp::min(m, n);
                if dim == 0 {
                    return Err(RlstError::MatrixIsEmpty((m as usize, n as usize)));
                }
                let mut ipiv = vec![0; dim as usize];
                let mut info = 0;
                unsafe {
                    $getrf(m, n, self.data_mut(), lda, &mut ipiv, &mut info);
                }

                match info {
                    0 => Ok(LuDecomposition { lu: self, ipiv }),
                    _ => Err(RlstError::LapackError(info)),
                }
            }
        }

        impl<ArrayImpl> ComputedLu for LuDecomposition<$scalar, ArrayImpl>
        where
            ArrayImpl: Shape<2> + Stride<2> + RawAccess<Item = $scalar>,
        {
            type ArrayImpl = ArrayImpl;

            type Item = $scalar;

            fn lu_data(&self) -> &LapackWrapper<Self::Item, Self::ArrayImpl> {
                &self.lu
            }

            fn ipiv(&self) -> &[i32] {
                &self.ipiv
            }

            fn solve_vec<OtherArrayImpl>(
                &self,
                trans: TransMode,
                rhs: &mut Array<OtherArrayImpl, 1>,
            ) -> RlstResult<()>
            where
                OtherArrayImpl: Shape<1> + Stride<1> + RawAccessMut<Item = Self::Item>,
            {
                let mut rhs_2d = rhs
                    .r_mut()
                    .insert_empty_axis(crate::dense::array::empty_axis::AxisPosition::Back);
                self.solve_mat(trans, &mut rhs_2d)
            }

            fn solve_mat<OtherArrayImpl>(
                &self,
                trans: TransMode,
                rhs: &mut Array<OtherArrayImpl, 2>,
            ) -> RlstResult<()>
            where
                OtherArrayImpl: Shape<2> + Stride<2> + RawAccessMut<Item = Self::Item>,
            {
                let (mb, nb, ldb) = lapack_dims(&rhs);

                let (m, n, lda) = self.lu.lapack_dims();

                assert_eq!(m, n, "Matrix must be square for LU solve.");
                assert_eq!(
                    mb, m,
                    "Right-hand side matrix row count does not match LU matrix."
                );

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
                        n,
                        nb,
                        self.lu.data(),
                        lda,
                        &self.ipiv,
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

            fn get_l(&self) -> DynArray<Self::Item, 2> {
                let (m, n, lda) = self.lu.lapack_dims();

                let k = std::cmp::min(m, n);

                let mut arr = DynArray::from_shape([m as usize, k as usize]);

                for col in 0..k as usize {
                    for row in col..m as usize {
                        if col == row {
                            unsafe { *arr.get_unchecked_mut([row, col]) = <$scalar as One>::one() };
                        } else {
                            unsafe {
                                *arr.get_unchecked_mut([row, col]) =
                                    *self.lu.data().get_unchecked(col * lda as usize + row);
                            };
                        }
                    }
                }

                arr
            }

            fn get_u(&self) -> DynArray<Self::Item, 2> {
                let (m, n, lda) = self.lu.lapack_dims();

                let k = std::cmp::min(m, n);

                let mut arr = DynArray::from_shape([k as usize, n as usize]);

                for col in 0..n as usize {
                    for row in 0..=std::cmp::min(col, k as usize - 1) {
                        unsafe {
                            *arr.get_unchecked_mut([row, col]) =
                                *self.lu.data().get_unchecked(col * lda as usize + row);
                        }
                    }
                }

                arr
            }

            fn get_perm(&self) -> Vec<usize> {
                let (m, _, _) = self.lu.lapack_dims();

                let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

                let mut perm = (0..m as usize).collect::<Vec<_>>();

                for (index, &elem) in ipiv.iter().enumerate() {
                    perm.swap(index, elem);
                }

                perm
            }

            fn get_p(&self) -> DynArray<Self::Item, 2> {
                let (m, _, _) = self.lu.lapack_dims();

                let mut arr = DynArray::from_shape([m as usize, m as usize]);

                let perm = self.get_perm();

                for col in 0..m as usize {
                    unsafe { *arr.get_unchecked_mut([perm[col], col]) = <$scalar as One>::one() };
                }

                arr
            }

            fn det(&self) -> Self::Item {
                let (m, n, lda) = self.lu.lapack_dims();
                assert_eq!(m, n, "Matrix must be square to compute determinant.");
                let mut det = self.lu.data()[0];
                if self.ipiv[0] != 1 {
                    det = -det;
                }
                for i in 1..m as usize {
                    det *= self.lu.data()[i * lda as usize + i];
                    if self.ipiv[i] != (i + 1) as i32 {
                        det = -det;
                    }
                }
                det
            }
        }
    };
}

implement_lu!(f64, dgetrf, dgetrs);
implement_lu!(f32, sgetrf, sgetrs);
implement_lu!(c64, zgetrf, zgetrs);
implement_lu!(c32, cgetrf, cgetrs);
