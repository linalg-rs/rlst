//! Lapack LU Decomposition.

use num::One;

use super::interface::getrf::Getrf;

use crate::dense::array::DynArray;
use crate::dense::linalg::lapack::interface::getrs::{Getrs, GetrsTransMode};
use crate::dense::linalg::traits::{
    GetDeterminant, GetL, GetLU, GetP, GetPermVec, GetU, Lu, Solve,
};
use crate::dense::types::TransMode;

use crate::dense::types::RlstResult;
use crate::{
    Array, BaseItem, FillFromResize, RawAccess, RawAccessMut, RlstScalar, Shape,
    UnsafeRandomAccessMut,
};

impl<Item, ArrayImpl> Lu for Array<ArrayImpl, 2>
where
    Item: Getrf + Clone + Default,
    ArrayImpl: BaseItem<Item = Item>,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
{
    type Output = LuDecomposition<Item>;

    fn lu(&self) -> RlstResult<Self::Output> {
        let mut lu_mat = DynArray::new_from(self);
        let (m, n, lda) = (lu_mat.shape()[0], lu_mat.shape()[1], lu_mat.shape()[0]);
        let mut ipiv = vec![0 as i32; std::cmp::min(m, n)];

        Item::getrf(m, n, lu_mat.data_mut(), lda, &mut ipiv)?;

        Ok(LuDecomposition { lu: lu_mat, ipiv })
    }
}

/// Store the result of an LU decomposition of a matrix.
pub struct LuDecomposition<Item> {
    lu: DynArray<Item, 2>,
    ipiv: Vec<i32>,
}

impl<Item, ArrayImpl> Solve<Array<ArrayImpl, 1>> for LuDecomposition<Item>
where
    Item: Getrs + Clone + Default,
    DynArray<Item, 1>: FillFromResize<Array<ArrayImpl, 1>>,
{
    type Output = DynArray<Item, 1>;

    fn solve(&self, trans: TransMode, b: &Array<ArrayImpl, 1>) -> RlstResult<Self::Output> {
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

        Item::getrs(
            trans,
            self.lu.shape()[0],
            1,
            self.lu.data(),
            self.lu.shape()[0],
            &self.ipiv,
            sol.data_mut(),
            ldb,
        )?;

        Ok(sol)
    }
}

impl<Item, ArrayImpl> Solve<Array<ArrayImpl, 2>> for LuDecomposition<Item>
where
    Item: Getrs + Clone + Default,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
{
    type Output = DynArray<Item, 2>;

    fn solve(&self, trans: TransMode, b: &Array<ArrayImpl, 2>) -> RlstResult<Self::Output> {
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

        Item::getrs(
            trans,
            self.lu.shape()[0],
            sol.shape()[1],
            self.lu.data(),
            self.lu.shape()[0],
            &self.ipiv,
            sol.data_mut(),
            ldb,
        )?;

        Ok(sol)
    }
}

impl<Item> GetL for LuDecomposition<Item>
where
    Item: Copy + Default + One,
{
    type Output = DynArray<Item, 2>;

    fn l_mat(&self) -> RlstResult<Self::Output> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut l_mat = DynArray::from_shape([m, k]);

        for col in 0..k as usize {
            for row in col..m as usize {
                if col == row {
                    unsafe { *l_mat.get_unchecked_mut([row, col]) = <Item as One>::one() };
                } else {
                    unsafe {
                        *l_mat.get_unchecked_mut([row, col]) =
                            *self.lu.data().get_unchecked(col * m as usize + row);
                    };
                }
            }
        }

        Ok(l_mat)
    }
}

impl<Item> GetU for LuDecomposition<Item>
where
    Item: Copy + Default + One,
{
    type Output = DynArray<Item, 2>;

    fn u_mat(&self) -> RlstResult<Self::Output> {
        let [m, n] = self.lu.shape();
        let k = std::cmp::min(m, n);
        let mut u_mat = DynArray::from_shape([k, n]);

        for col in 0..n {
            for row in 0..=std::cmp::min(col, k - 1) {
                unsafe {
                    *u_mat.get_unchecked_mut([row, col]) =
                        *self.lu.data().get_unchecked(col * m as usize + row);
                }
            }
        }

        Ok(u_mat)
    }
}

impl<Item> GetLU for LuDecomposition<Item>
where
    Item: Copy + Default + One,
{
    type Output = (DynArray<Item, 2>, DynArray<Item, 2>);

    fn l_u_mat(&self) -> RlstResult<Self::Output> {
        let l_mat = self.l_mat()?;
        let u_mat = self.u_mat()?;
        Ok((l_mat, u_mat))
    }
}

impl<Item> GetPermVec for LuDecomposition<Item> {
    fn perm_vec(&self) -> RlstResult<Vec<usize>> {
        let ipiv: Vec<usize> = self.ipiv.iter().map(|&elem| (elem as usize) - 1).collect();

        let mut perm = (0..self.lu.shape()[0]).collect::<Vec<_>>();

        for (index, &elem) in ipiv.iter().enumerate() {
            perm.swap(index, elem);
        }

        Ok(perm)
    }
}

impl<Item> GetP for LuDecomposition<Item>
where
    Item: Copy + Default + One,
{
    type Output = DynArray<Item, 2>;

    fn p_mat(&self) -> RlstResult<Self::Output> {
        let m = self.lu.shape()[0];
        let mut p_mat = DynArray::from_shape([m, m]);

        let perm = self.perm_vec()?;

        for col in 0..m {
            unsafe { *p_mat.get_unchecked_mut([perm[col], col]) = <Item as One>::one() };
        }

        Ok(p_mat)
    }
}

impl<Item> GetDeterminant for LuDecomposition<Item>
where
    Item: RlstScalar,
{
    type Item = Item;

    fn det(&self) -> Item {
        let [m, n] = self.lu.shape();
        assert_eq!(m, n, "Matrix must be square to compute determinant.");
        let mut det = self.lu.data()[0];
        if self.ipiv[0] != 1 {
            det = -det;
        }
        for i in 1..m as usize {
            det *= unsafe { *self.lu.data().get_unchecked(i * m as usize + i) };
            if self.ipiv[i] != (i + 1) as i32 {
                det = -det;
            }
        }
        det
    }
}
