//! Implement Cholesky decomposition and solve for positive definite matrices.

use crate::{
    base_types::{RlstError, RlstResult, UpLo},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::{posv::PosvUplo, potrf::PotrfUplo},
    },
    traits::{
        accessors::{RawAccessMut, UnsafeRandomAccessMut},
        array::{BaseItem, FillFromResize, Shape},
        linalg::{
            decompositions::{Cholesky, CholeskySolve},
            lapack::Lapack,
        },
    },
};

impl<Item, ArrayImpl> Cholesky for Array<ArrayImpl, 2>
where
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    Item: Lapack,
    DynArray<<ArrayImpl as BaseItem>::Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
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
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    RhsArrayImpl: BaseItem<Item = Item> + Shape<NDIM>,
    Item: Lapack,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
    DynArray<Item, NDIM>: FillFromResize<Array<RhsArrayImpl, NDIM>>,
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
