//! Implement the triangular solver trait.

use crate::{
    UnsafeRandom1DAccessByValue,
    base_types::{RlstResult, UpLo},
    dense::{
        array::{Array, DynArray},
        linalg::lapack::interface::trsm::{TrsmDiag, TrsmSide, TrsmTransA, TrsmUplo},
    },
    traits::{
        base_operations::Shape,
        linalg::{decompositions::SolveTriangular, lapack::Lapack},
    },
};

impl<Item, ArrayImpl, RhsArrayImpl, const NDIM: usize> SolveTriangular<Array<RhsArrayImpl, NDIM>>
    for Array<ArrayImpl, 2>
where
    Item: Lapack,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    RhsArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<NDIM>,
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

        let nrhs = if NDIM == 1 {
            1
        } else if NDIM == 2 {
            rhs.shape()[1]
        } else {
            panic!("The right-hand side must be one or two dimensional: NDIM = {NDIM}");
        };

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
            a.data_mut().unwrap(),
            m,
            b.data_mut().unwrap(),
            m,
        )?;

        Ok(b)
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use num::Zero;
    use paste::paste;

    macro_rules! implement_triangular_solve_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<test_solve_triangular_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_equally_distributed(0);

                for row in 0..n {
                    for col in 1 + row..n {
                        a[[row, col]] = <$scalar>::zero(); // Make it lower triangular
                    }
                }

                let mut x_actual = DynArray::<$scalar, 2>::from_shape([n, 1]);
                x_actual.fill_from_seed_equally_distributed(1);

                let b = dot!(a.r(), x_actual.r());

                let x = a.solve_triangular(UpLo::Lower, &b).unwrap();

                crate::assert_array_relative_eq!(x_actual, x, $tol);
            }


                    }
        };
    }

    implement_triangular_solve_test!(f32, 5E-3);
    implement_triangular_solve_test!(f64, 1E-10);
    implement_triangular_solve_test!(c32, 5E-3);
    implement_triangular_solve_test!(c64, 1E-10);
}
