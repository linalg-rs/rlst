//! Implement Cholesky decomposition and solve for positive definite matrices.

use crate::{
    UnsafeRandom1DAccessByValue,
    base_types::{RlstError, RlstResult, UpLo},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::{posv::PosvUplo, potrf::PotrfUplo},
    },
    traits::{
        base_operations::Shape,
        linalg::{
            decompositions::{Cholesky, CholeskySolve},
            lapack::Lapack,
        },
    },
};

impl<Item, ArrayImpl> Cholesky for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    type Item = Item;

    fn cholesky(&self, uplo: UpLo) -> RlstResult<DynArray<Self::Item, 2>> {
        let [m, n] = self.shape();

        assert_eq!(
            m, n,
            "Cholesky decomposition requires a square matrix, but got shape: {m}x{n}"
        );

        let mut a = DynArray::new_from(self);

        let uplo = match uplo {
            UpLo::Upper => PotrfUplo::Upper,
            UpLo::Lower => PotrfUplo::Lower,
        };

        Item::potrf(uplo, m, a.data_mut(), m)?;

        // We manually set the lower or upper part of the matrix to zero.

        if uplo == PotrfUplo::Upper {
            for col in 0..n {
                for row in (col + 1)..m {
                    unsafe { *a.get_unchecked_mut([row, col]) = Item::zero() };
                }
            }
        } else {
            for col in 0..n {
                for row in 0..col {
                    unsafe { *a.get_unchecked_mut([row, col]) = Item::zero() };
                }
            }
        }

        Ok(a)
    }
}

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> CholeskySolve<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    RhsArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
    Item: Lapack,
{
    type Output = DynArray<Item, NDIM>;

    fn cholesky_solve(
        &self,
        uplo: UpLo,
        rhs: &Array<RhsArrayImpl, NDIM>,
    ) -> RlstResult<Self::Output> {
        if NDIM > 2 {
            return Err(RlstError::GeneralError(
                "The right-hand side cannot have more than two dimensions.".to_string(),
            ));
        }

        let [m, n] = self.shape();

        assert_eq!(
            m, n,
            "Cholesky solve requires a square matrix, but got shape: {m}x{n}"
        );

        assert_eq!(
            m,
            rhs.shape()[0],
            "Left-hand side and right-hand side must have the same number of rows. {} != {}",
            m,
            rhs.shape()[0]
        );

        let mut a = DynArray::new_from(self);

        let nrhs = if NDIM == 1 { 1 } else { rhs.shape()[1] };

        let mut b = DynArray::new_from(rhs);

        let ldb = b.shape()[0];

        let uplo = match uplo {
            UpLo::Upper => PosvUplo::Upper,
            UpLo::Lower => PosvUplo::Lower,
        };

        Item::posv(uplo, m, nrhs, a.data_mut(), m, b.data_mut(), ldb)?;

        Ok(b)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::base_operations::*;
    use paste::paste;

    macro_rules! implement_cholesky_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_cholesky_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_normally_distributed(0);

                // Make it symmetric positive definite
                a = dot!(a.r().conj().transpose().eval(), a.r());

                let z = a.cholesky(UpLo::Upper).unwrap();

                let actual = dot!(z.r().conj().transpose().eval(), z.r());

                crate::assert_array_relative_eq!(actual, a, $tol);

                // Now solve a linear system with Cholesky

                let mut x_expected = DynArray::<$scalar, 2>::from_shape([n, 2]);
                x_expected.fill_from_seed_equally_distributed(1);

                let b = dot!(a.r(), x_expected.r());

                let x_actual = a.cholesky_solve(UpLo::Upper, &b).unwrap();

                crate::assert_array_relative_eq!(x_actual, x_expected, $tol);
            }

                    }
        };
    }

    implement_cholesky_test!(f32, 1E-3);
    implement_cholesky_test!(f64, 1E-10);
    implement_cholesky_test!(c32, 1E-3);
    implement_cholesky_test!(c64, 1E-10);
}
