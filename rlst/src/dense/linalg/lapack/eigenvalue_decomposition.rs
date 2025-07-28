//! Implementation of the Schur and eigenvalue decomposition for general matrices.

use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::geev::{JobVl, JobVr};
use crate::traits::accessors::RawAccessMut;
use crate::traits::array::{BaseItem, FillFromResize, Shape};
use crate::traits::linalg::decompositions::EigenvalueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;

use super::interface::gees::JobVs;

/// Symmetric eigenvalue decomposition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigMode {
    /// Compute eigenvalues only.
    EigenvaluesOnly,
    /// Compute right eigenvectors
    RightEigenvectors,
    /// Compute left eigenvectors
    LeftEigenvectors,
    /// Compute both right and left eigenvectors
    BothEigenvectors,
}

impl<Item, ArrayImpl> EigenvalueDecomposition for Array<ArrayImpl, 2>
where
    ArrayImpl: BaseItem<Item = Item> + Shape<2>,
    Item: Lapack,
    DynArray<Item, 2>: FillFromResize<Array<ArrayImpl, 2>>,
{
    type Item = Item;

    fn eigenvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Complex, 1>> {
        let mut a = DynArray::new_from(self);

        let [m, n] = a.shape();
        assert_eq!(
            m, n,
            "Matrix must be square for eigenvalue decomposition. Got shape: [{m}, {n}]"
        );

        let mut w = DynArray::from_shape([n]);

        Item::gees(JobVs::None, n, a.data_mut(), n, w.data_mut(), None, 1)?;

        Ok(w)
    }

    fn schur(&self) -> RlstResult<(DynArray<Self::Item, 2>, DynArray<Self::Item, 2>)> {
        let mut a = DynArray::new_from(self);

        let [m, n] = a.shape();
        assert_eq!(
            m, n,
            "Matrix must be square for Schur decomposition. Got shape: [{m}, {n}]"
        );

        let mut w = DynArray::from_shape([n]);

        let mut vs = DynArray::from_shape([n, n]);

        Item::gees(
            JobVs::Compute,
            n,
            a.data_mut(),
            n,
            w.data_mut(),
            Some(vs.data_mut()),
            n,
        )?;

        Ok((a, vs))
    }

    fn eig(
        &self,
        mode: EigMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Complex, 1>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
        Option<DynArray<<Self::Item as RlstScalar>::Complex, 2>>,
    )> {
        let mut a = DynArray::new_from(self);

        let [m, n] = a.shape();
        assert_eq!(
            m, n,
            "Matrix must be square for eigenvalue decomposition. Got shape: [{m}, {n}]"
        );

        let mut w = DynArray::<<Item as RlstScalar>::Complex, 1>::from_shape([n]);

        let (jobvl, jobvr, mut vl, mut vr) = match mode {
            EigMode::EigenvaluesOnly => (JobVl::None, JobVr::None, None, None),
            EigMode::RightEigenvectors => (
                JobVl::None,
                JobVr::Compute,
                None,
                Some(DynArray::<Item::Complex, 2>::from_shape([n, n])),
            ),
            EigMode::LeftEigenvectors => (
                JobVl::Compute,
                JobVr::None,
                Some(DynArray::<Item::Complex, 2>::from_shape([n, n])),
                None,
            ),
            EigMode::BothEigenvectors => (
                JobVl::Compute,
                JobVr::Compute,
                Some(DynArray::<Item::Complex, 2>::from_shape([n, n])),
                Some(DynArray::<Item::Complex, 2>::from_shape([n, n])),
            ),
        };

        Item::geev(
            jobvl,
            jobvr,
            n,
            a.data_mut(),
            n,
            w.data_mut(),
            vl.as_mut().map(|v| v.data_mut()),
            n,
            vr.as_mut().map(|v| v.data_mut()),
            n,
        )?;

        Ok((w, vr, vl))
    }
}
