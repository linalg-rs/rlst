//! Implementation of the singular value decomposition using LAPACK.

use crate::base_types::RlstResult;
use crate::dense::array::{Array, DynArray};
use crate::dense::linalg::lapack::interface::gesdd::JobZ;
use crate::traits::base_operations::Shape;
use crate::traits::linalg::base::Gemm;
use crate::traits::linalg::decompositions::SingularvalueDecomposition;
use crate::traits::linalg::lapack::Lapack;
use crate::traits::rlst_num::RlstScalar;
use crate::UnsafeRandom1DAccessByValue;

/// Symmetric eigenvalue decomposition mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvdMode {
    /// Compute full matrices U and V.
    Full,
    /// Compute compact matrices U and V.
    Compact,
}

impl<Item, ArrayImpl> SingularvalueDecomposition for Array<ArrayImpl, 2>
where
    Item: Lapack + Gemm,
    ArrayImpl: UnsafeRandom1DAccessByValue<Item = Item> + Shape<2>,
{
    type Item = Item;

    fn singularvalues(&self) -> RlstResult<DynArray<<Self::Item as RlstScalar>::Real, 1>> {
        let mut a = DynArray::new_from(self);
        let [m, n] = a.shape();
        let k = std::cmp::min(m, n);

        let mut s = DynArray::<<Self::Item as RlstScalar>::Real, 1>::from_shape([k]);

        Item::gesdd(
            JobZ::N,
            m,
            n,
            a.data_mut(),
            m,
            s.data_mut(),
            None,
            1,
            None,
            1,
        )?;

        Ok(s)
    }

    fn svd(
        &self,
        mode: SvdMode,
    ) -> RlstResult<(
        DynArray<<Self::Item as RlstScalar>::Real, 1>,
        DynArray<Self::Item, 2>,
        DynArray<Self::Item, 2>,
    )> {
        let mut a = DynArray::new_from(self);
        let [m, n] = a.shape();
        let k = std::cmp::min(m, n);
        let mut s = DynArray::<<Self::Item as RlstScalar>::Real, 1>::from_shape([k]);
        let (mut u, mut vt, ldvt) = match mode {
            SvdMode::Full => (
                DynArray::<Self::Item, 2>::from_shape([m, m]),
                DynArray::<Self::Item, 2>::from_shape([n, n]),
                n,
            ),
            SvdMode::Compact => (
                DynArray::<Self::Item, 2>::from_shape([m, k]),
                DynArray::<Self::Item, 2>::from_shape([k, n]),
                k,
            ),
        };

        let jobz = match mode {
            SvdMode::Full => JobZ::A,
            SvdMode::Compact => JobZ::S,
        };

        Item::gesdd(
            jobz,
            m,
            n,
            a.data_mut(),
            m,
            s.data_mut(),
            Some(u.data_mut()),
            m,
            Some(vt.data_mut()),
            ldvt,
        )?;

        Ok((s, u, vt))
    }
}
