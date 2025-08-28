//! Implementation of the Schur and eigenvalue decomposition for general matrices.

use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::geev::{JobVl, JobVr};
use crate::traits::base_operations::Shape;
use crate::traits::linalg::decompositions::EigenvalueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
use crate::UnsafeRandom1DAccessByValue;

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
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
    Item: Lapack,
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

#[cfg(test)]
mod test {

    use super::*;
    use crate::base_types::{c32, c64};
    use crate::dense::array::DynArray;
    use crate::dot;
    use crate::traits::base_operations::*;
    use crate::traits::linalg::Inverse;
    use crate::traits::linalg::SymmEig;
    use itertools::izip;
    use num::Zero;
    use paste::paste;

    macro_rules! implement_eigendecomposition_tests {
        ($scalar:ty, $tol:expr) => {
            paste! {
            #[test]
            fn [<test_eigendecomposition_$scalar>]() {
                let n = 11;
                let mut a = DynArray::<$scalar, 2>::from_shape([n, n]);
                a.fill_from_seed_equally_distributed(0);

                let lam = a.eigenvalues().unwrap();

                let (t, z) = a.schur().unwrap();

                // Test the Schur decomposition

                let actual = dot!(z.r(), t.r(), z.r().conj().transpose().eval());

                crate::assert_array_relative_eq!(actual, a, $tol);

                let (lam2, vr, vl) = a.eig(EigMode::BothEigenvectors).unwrap();

                crate::assert_array_relative_eq!(lam, lam2, $tol);

                // Test the left eigenvectors

                // First convert a to a complex matrix
                let a_complex = a.into_type::<<$scalar as RlstScalar>::Complex>().eval();

                // Now create a diagonal matrix from the eigenvalues

                let mut diag = DynArray::from_shape([n, n]);

                izip!(diag.diag_iter_mut(), lam2.iter_value()).for_each(|(v_elem, w_elem)| {
                    *v_elem = w_elem;
                });

                // Now check the left eigenvectors

                let vlh = vl.unwrap().conj().transpose().eval();

                let actual = dot!(vlh.inverse().unwrap(), dot!(diag.r(), vlh));

                // We test the absolute distance since some imaginary parts are zero
                // making relative tests fail.

                crate::assert_array_abs_diff_eq!(actual, a_complex, $tol);

                // Now check the right eigenvectors

                let vr = vr.unwrap();

                let actual = dot!(vr.r(), dot!(diag, vr.r().inverse().unwrap()));
                crate::assert_array_abs_diff_eq!(actual, a_complex, $tol);

            }
            }
        };
    }

    implement_eigendecomposition_tests!(f32, 1E-4);
    implement_eigendecomposition_tests!(f64, 1E-10);
    implement_eigendecomposition_tests!(c32, 5E-3);
    implement_eigendecomposition_tests!(c64, 1E-10);
}
