//! Lapack LU Decomposition.

use num::One;

use crate::base_types::{RlstError, RlstResult, TransMode};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::getrs::GetrsTransMode;
use crate::traits::accessors::{RawAccess, RawAccessMut, UnsafeRandomAccessMut};
use crate::traits::array::{BaseItem, FillFromResize, Shape};
use crate::traits::linalg::decompositions::Lu;
use crate::traits::linalg::lapack::Lapack;

impl<Item, ArrayImpl> Lu for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: BaseItem<Item = Item>,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
{
    type Item = Item;
    fn lu(&self) -> RlstResult<LuDecomposition<Item>> {
        let mut lu_mat = DynArray::new_from(self);
        let (m, n, lda) = (lu_mat.shape()[0], lu_mat.shape()[1], lu_mat.shape()[0]);
        let mut ipiv = vec![0_i32; std::cmp::min(m, n)];

        Item::getrf(m, n, lu_mat.data_mut(), lda, &mut ipiv)?;

        Ok(LuDecomposition { lu: lu_mat, ipiv })
    }
}

/// Store the result of an LU decomposition of a matrix.
/// The LU Decomposition is represented as `A = P * L * U`, where:
/// - `A` is the original matrix,
/// - `P` is a permutation matrix,
/// - `L` is a lower triangular matrix with unit diagonal,
/// - `U` is an upper triangular matrix.
pub struct LuDecomposition<Item> {
    lu: DynArray<Item, 2>,
    ipiv: Vec<i32>,
}

impl<Item> LuDecomposition<Item>
where
    Item: Lapack,
{
    /// Return the solution of `x` of the linear system `Ax = b`.
    pub fn solve<ArrayImpl, const NDIM: usize>(
        &self,
        trans: TransMode,
        b: &Array<ArrayImpl, NDIM>,
    ) -> RlstResult<DynArray<Item, NDIM>>
    where
        DynArray<Item, NDIM>: FillFromResize<Array<ArrayImpl, NDIM>>,
    {
        if NDIM > 2 {
            return Err(RlstError::GeneralError(
                "LU solve is only implemented for 1D and 2D arrays.".to_string(),
            ));
        }

        let mut sol = DynArray::new_from(b);

        assert_eq!(
            self.lu.shape()[0],
            self.lu.shape()[1],
            "Matrix must be square for LU solve."
        );
        assert_eq!(
            self.lu.shape()[0],
            sol.shape()[0],
            "Right-hand side vector length does not match LU matrix size."
        );

        let trans = match trans {
            TransMode::NoTrans => GetrsTransMode::NoTranspose,
            TransMode::Trans => GetrsTransMode::Transpose,
            TransMode::ConjTrans => GetrsTransMode::ConjugateTranspose,
            _ => panic!("Transposition mode not supported for LU solve."),
        };

        let ldb = sol.shape()[0];

        let nrhs = if NDIM == 1 { 1 } else { sol.shape()[1] };

        Item::getrs(
            trans,
            self.lu.shape()[0],
            nrhs,
            self.lu.data(),
            self.lu.shape()[0],
            &self.ipiv,
            sol.data_mut(),
            ldb,
        )?;

        Ok(sol)
    }

    /// Return the L matrix of `A = P * L * U`.
    pub fn l_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut l_mat = DynArray::from_shape([m, k]);

        for col in 0..k as usize {
            for row in col..m {
                if col == row {
                    unsafe { *l_mat.get_unchecked_mut([row, col]) = <Item as One>::one() };
                } else {
                    unsafe {
                        *l_mat.get_unchecked_mut([row, col]) =
                            *self.lu.data().get_unchecked(col * m + row);
                    };
                }
            }
        }

        Ok(l_mat)
    }

    /// Return the U matrix of `A = P * L * U`.
    pub fn u_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut u_mat = DynArray::from_shape([k, n]);

        for col in 0..n {
            for row in 0..=std::cmp::min(col, k - 1) {
                unsafe {
                    *u_mat.get_unchecked_mut([row, col]) =
                        *self.lu.data().get_unchecked(col * m + row);
                }
            }
        }

        Ok(u_mat)
    }

    /// Return the P matrix of `A = P * L * U`.
    pub fn p_mat(&self) -> RlstResult<DynArray<Item, 2>> {
        let [m, _] = self.lu.shape();
        let mut p_mat = DynArray::from_shape([m, m]);

        let perm = self.perm_vec()?;

        for (i, &j) in perm.iter().enumerate() {
            unsafe {
                *p_mat.get_unchecked_mut([j, i]) = <Item as One>::one();
            }
        }

        Ok(p_mat)
    }

    /// Return the permutation vector that defines P
    /// If `perm[i] = j`, then the `i`-th row of the LU decomposition corresponds to the `j`-th row of
    /// the original matrix.
    pub fn perm_vec(&self) -> RlstResult<Vec<usize>> {
        let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

        let mut perm = (0..self.lu.shape()[0]).collect::<Vec<_>>();

        for (index, &elem) in ipiv.iter().enumerate() {
            perm.swap(index, elem);
        }

        Ok(perm)
    }

    /// Return the determinat of the matrix `A`.
    pub fn det(&self) -> Item {
        let [m, n] = self.lu.shape();
        assert_eq!(m, n, "Matrix must be square to compute determinant.");
        let mut det = self.lu.data()[0];
        if self.ipiv[0] != 1 {
            det = -det;
        }
        for i in 1..m {
            det *= unsafe { *self.lu.data().get_unchecked(i * m + i) };
            if self.ipiv[i] != (i + 1) as i32 {
                det = -det;
            }
        }
        det
    }
}
