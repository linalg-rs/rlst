//! Implementation of the symmetric eigenvalue decomposition using LAPACK.

use crate::base_types::{RlstResult, UpLo};
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::ev::{self, Ev, EvUplo};
use crate::traits::accessors::RawAccessMut;
use crate::traits::base_operations::{BaseItem, FillFromResize, Shape};
use crate::traits::linalg::decompositions::SymmEig;
use crate::traits::linalg::lapack::Lapack;

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
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    Item: Lapack,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
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
