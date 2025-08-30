//! Implementation of the symmetric eigenvalue decomposition using LAPACK.

use crate::base_types::{RlstResult, UpLo};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::ev::{self, Ev, EvUplo};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::SymmEig;
use crate::traits::linalg::lapack::Lapack;
use crate::UnsafeRandom1DAccessByValue;

/// Symmetric eigenvalue decomposition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmEigMode {
    /// Compute the eigenvalues only.
    EigenvaluesOnly,
    /// Compute the eigenvalues and eigenvectors.
    EigenvaluesAndEigenvectors,
}

impl<Item, ArrayImpl> SymmEig for Array<ArrayImpl, 2>
where
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
{
    type Item = Item;

    fn eigh(
        &self,
        uplo: UpLo,
        mode: SymmEigMode,
    ) -> RlstResult<(DynArray<Item::Real, 1>, Option<DynArray<Item, 2>>)> {
        let m = self.shape()[0];
        let n = self.shape()[1];
        assert_eq!(
            m, n,
            "Matrix must be square for symmetric eigenvalue decomposition."
        );

        let mut a = DynArray::new_from(self);

        let mut w = DynArray::from_shape([n]);

        let uplo = match uplo {
            UpLo::Upper => EvUplo::Upper,
            UpLo::Lower => EvUplo::Lower,
        };

        let jobz = match mode {
            SymmEigMode::EigenvaluesOnly => ev::JobZEv::None,
            SymmEigMode::EigenvaluesAndEigenvectors => ev::JobZEv::Compute,
        };

        <Item as Ev>::ev(jobz, uplo, n, a.data_mut(), n, w.data_mut())?;

        match mode {
            SymmEigMode::EigenvaluesOnly => Ok((w, None)),
            SymmEigMode::EigenvaluesAndEigenvectors => Ok((w, Some(a))),
        }
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;

    use crate::RlstScalar;
    use itertools::izip;
    use paste::paste;

    macro_rules! implement_symm_eig_test {
        ($scalar:ty, $tol:expr) => {
            paste! {

            #[test]
            fn [<symm_eig_test_$scalar>]() {
                let n = 10;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_equally_distributed(0);

                let a = DynArray::new_from(&(a.r() + a.r().conj().transpose()));

                let (w1, _) = a
                    .eigh(UpLo::Upper, SymmEigMode::EigenvaluesOnly)
                    .unwrap();

                let (w2, v) = a
                    .eigh(UpLo::Upper, SymmEigMode::EigenvaluesAndEigenvectors)
                    .unwrap();

                let v = v.unwrap();

                crate::assert_array_relative_eq!(w1, w2, $tol);

                let mut lambda = DynArray::<$scalar, 2>::from_shape([n, n]);

                izip!(lambda.diag_iter_mut(), w1.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = RlstScalar::from_real(w_elem);
                });

                let vt = DynArray::new_from(
                    &v.r().conj().transpose(),
                );

                let actual = dot!(v.r(), dot!(lambda.r(), vt.r()));

                crate::assert_array_relative_eq!(actual, a, $tol);
            }

                    }
        };
    }

    implement_symm_eig_test!(f32, 1E-4);
    implement_symm_eig_test!(f64, 1E-10);
    implement_symm_eig_test!(c32, 1E-4);
    implement_symm_eig_test!(c64, 1E-10);
}
