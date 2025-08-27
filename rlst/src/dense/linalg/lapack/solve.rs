//! Implementation of the linear solver trait.

use crate::{
    base_types::{RlstError, RlstResult, TransMode},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::gels::GelsTransMode,
    },
    traits::{
        accessors::UnsafeRandom1DAccessByValue,
        base_operations::{EvaluateObject, Shape},
        linalg::{
            decompositions::{Lu, Solve},
            lapack::Lapack,
        },
    },
};

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> Solve<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    RhsArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
{
    type Output = DynArray<Item, NDIM>;

    fn solve(&self, rhs: &Array<RhsArrayImpl, NDIM>) -> RlstResult<Self::Output> {
        if NDIM > 2 {
            return Err(RlstError::GeneralError(
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
            Ok(self.lu()?.solve(TransMode::NoTrans, rhs)?)
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
                .fill_from(rhs);

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

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::empty_array;
    use crate::traits::base_operations::*;
    use crate::Max;
    use crate::MultIntoResize;
    use crate::RlstScalar;
    use paste::paste;

    macro_rules! implement_test_solve {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            pub fn [<test_solve_square_$scalar>]() {
                let m = 5;
                let n = 5;
                let nrhs = 4;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let mut x_expected = DynArray::<$scalar, 2>::from_shape([n, nrhs]);
                x_expected.fill_from_seed_equally_distributed(1);

                let rhs = dot!(a.r(), x_expected.r());

                let x_actual = a.solve(&rhs).unwrap();

                crate::assert_array_relative_eq!(x_actual, x_expected, $tol);
            }

            #[test]
            pub fn [<test_solve_thin_$scalar>]() {
                let m = 10;
                let n = 5;
                let nrhs = 4;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let mut x_expected = DynArray::<$scalar, 2>::from_shape([n, nrhs]);
                x_expected.fill_from_seed_equally_distributed(1);

                let rhs = dot!(a.r(), x_expected.r());

                let x_actual = a.solve(&rhs).unwrap();

                crate::assert_array_relative_eq!(x_actual, x_expected, $tol);
            }

            #[test]
            pub fn [<test_solve_thick_$scalar>]() {
                let m = 5;
                let n = 10;
                let nrhs = 4;
                let mut a = DynArray::<$scalar, 2>::from_shape([m, n]);
                a.fill_from_seed_equally_distributed(0);

                let mut rhs = DynArray::<$scalar, 2>::from_shape([m, nrhs]);
                rhs.fill_from_seed_equally_distributed(1);

                let x_actual = a.solve(&rhs).unwrap();

                let max_res = (dot!(a.r(), x_actual.r()) - rhs.r())
                    .iter_value()
                    .map(|v| crate::RlstScalar::abs(v))
                    .fold(0.0, |acc, v| Max::max(acc, v));

                assert!(max_res < $tol);
            }

                    }
        };
    }

    implement_test_solve!(f32, 1E-4);
    implement_test_solve!(f64, 1E-10);
    implement_test_solve!(c32, 1E-4);
    implement_test_solve!(c64, 1E-10);
}
