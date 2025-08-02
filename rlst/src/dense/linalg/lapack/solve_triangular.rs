//! Implement the triangular solver trait.

use crate::{
    base_types::{RlstResult, UpLo},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::trsm::{TrsmDiag, TrsmSide, TrsmTransA, TrsmUplo},
    },
    traits::{
        accessors::RawAccessMut,
        base_operations::{BaseItem, FillFromResize, Shape},
        linalg::{decompositions::SolveTriangular, lapack::Lapack},
    },
};

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> SolveTriangular<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    RhsArrayImpl: BaseItem<Item = Item> + Shape<NDIM>,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
    DynArray<Item, NDIM>: FillFromResize<Array<RhsArrayImpl, NDIM>>,
{
    type Output = DynArray<Item, NDIM>;

    fn solve_triangular(
        &self,
        uplo: UpLo,
        rhs: &Array<RhsArrayImpl, NDIM>,
    ) -> RlstResult<Self::Output> {
        let [m, n] = self.shape();

        assert_eq!(
            m, n,
            "Triangular solver requires a square matrix, but got shape: {m}x{n}"
        );

        assert_eq!(
            m,
            rhs.shape()[0],
            "Left-hand side and right-hand side must have the same number of rows. {} != {}",
            m,
            rhs.shape()[0]
        );

        let nrhs = rhs.shape()[1];

        let mut a = DynArray::new_from(self);
        let mut b = DynArray::new_from(rhs);

        let uplo = match uplo {
            UpLo::Upper => TrsmUplo::Upper,
            UpLo::Lower => TrsmUplo::Lower,
        };

        Item::trsm(
            TrsmSide::Left,
            uplo,
            TrsmTransA::NoTrans,
            TrsmDiag::NonUnit,
            m,
            nrhs,
            Item::one(),
            a.data_mut(),
            m,
            b.data_mut(),
            m,
        )?;

        Ok(b)
    }
}
