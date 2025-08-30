//! Lapack LU Decomposition.

use num::One;

use crate::base_types::{RlstError, RlstResult, TransMode};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::getrs::GetrsTransMode;
use crate::traits::linalg::decompositions::Lu;
use crate::traits::linalg::lapack::Lapack;
use crate::{Shape, UnsafeRandom1DAccessByValue};

impl<Item, ArrayImpl> Lu for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
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
        ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
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

        for col in 0..k {
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;

    use crate::empty_array;
    use crate::MultIntoResize;
    use crate::RlstScalar;
    use paste::paste;

    macro_rules! impl_lu_tests {

        ($scalar:ty, $tol:expr) => {
            paste! {
                #[test]
                fn [<test_lu_thick_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([8, 20]);

                    arr.fill_from_seed_normally_distributed(0);

                    let lu = DynArray::new_from(&arr).lu().unwrap();


                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        crate::empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_square_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 12]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res =
                        empty_array().simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_lu_solve_$scalar>]() {
                    let dim = [12, 12];
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 12]);
                    arr.fill_from_seed_equally_distributed(0);
                    let mut x_expected = DynArray::<$scalar, 1>::from_shape([dim[0]]);
                    x_expected.fill_from_seed_equally_distributed(1);
                    let rhs = empty_array().simple_mult_into_resize(arr.r(), x_expected.r());

                    let x_actual = arr.lu().unwrap().solve(TransMode::NoTrans, &rhs).unwrap();

                    crate::assert_array_relative_eq!(x_actual, x_expected, $tol)
                }



                #[test]
                fn [<test_lu_thin_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([12, 8]);

                    arr.fill_from_seed_normally_distributed(0);
                    let arr2 = DynArray::new_from(&arr);

                    let lu = arr2.lu().unwrap();

                    let l_mat = lu.l_mat().unwrap();
                    let u_mat = lu.u_mat().unwrap();
                    let p_mat = lu.p_mat().unwrap();

                    let res = empty_array::<$scalar, 2>();

                    let res =
                        res.simple_mult_into_resize(empty_array().simple_mult_into_resize(p_mat, l_mat), u_mat);

                    crate::assert_array_relative_eq!(res, arr, $tol)
                }

                #[test]
                fn [<test_det_$scalar>]() {
                    let mut arr = DynArray::<$scalar, 2>::from_shape([2, 2]);
                    arr[[0, 1]] = $scalar::from_real(3.0);
                    arr[[1, 0]] = $scalar::from_real(2.0);

                    let det = arr.lu().unwrap().det();

                    approx::assert_relative_eq!(det, $scalar::from_real(-6.0), epsilon=$tol);
                }



            }
        };
    }

    impl_lu_tests!(f64, 1E-12);
    impl_lu_tests!(f32, 1E-4);
    impl_lu_tests!(c64, 1E-12);
    impl_lu_tests!(c32, 1E-4);
}
