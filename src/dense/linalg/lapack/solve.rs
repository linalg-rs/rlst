//! Implementation of the linear solver trait.

use crate::{
    dense::{
        array::DynArray,
        linalg::{
            lapack::interface::gels::GelsTransMode,
            traits::{Lu, Solve},
        },
    },
    Array, BaseItem, EvaluateArray, FillFrom, FillFromResize, RawAccessMut, Shape,
    UnsafeRandom1DAccessByValue,
};

use super::interface::Lapack;

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> Solve<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    RhsArrayImpl: BaseItem<Item = Item> + Shape<NDIM>,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
    DynArray<Item, NDIM>: FillFromResize<Array<RhsArrayImpl, NDIM>>,
    RhsArrayImpl: BaseItem<Item = Item> + UnsafeRandom1DAccessByValue<Item = Item>,
{
    type Output = DynArray<Item, NDIM>;

    fn solve(
        &self,
        rhs: &Array<RhsArrayImpl, NDIM>,
    ) -> crate::dense::types::RlstResult<Self::Output> {
        if NDIM > 2 {
            return Err(crate::dense::types::RlstError::GeneralError(
                "The right-hand side cannot have more than two dimensions.".to_string(),
            ));
        }

        let [m, n] = self.shape();

        assert_eq!(
            m,
            rhs.shape()[0],
            "Left-hand side and right-hand side must have the same number of rows. {} != {}",
            m,
            rhs.shape()[0]
        );

        if m == n {
            // Square matrix case
            Ok(self
                .lu()?
                .solve(crate::dense::types::TransMode::NoTrans, &rhs)?)
        } else {
            // Rectangular matrix case

            let mut extended_shape = rhs.shape();
            // The right-hand side must have enough rows to fit solution vectors
            // in the case that n > m.
            extended_shape[0] = std::cmp::max(1, std::cmp::max(m, n));
            let ldrhs = extended_shape[0];

            let mut new_rhs = DynArray::<Item, NDIM>::from_shape(extended_shape);
            new_rhs
                .r_mut()
                .into_subview([0; NDIM], rhs.shape())
                .fill_from(&rhs);

            let n_rhs = if NDIM == 1 { 1 } else { rhs.shape()[1] };

            let mut a = DynArray::new_from(self);

            Item::gels(
                GelsTransMode::NoTranspose,
                m,
                n,
                n_rhs,
                a.data_mut(),
                m,
                new_rhs.data_mut(),
                ldrhs,
            )?;

            let mut output_shape = [0; NDIM];
            output_shape[0] = n;
            if NDIM == 2 {
                output_shape[1] = n_rhs;
            }

            let output = new_rhs.r().into_subview([0; NDIM], output_shape).eval();
            Ok(output)
        }
    }
}
