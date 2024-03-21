//! Sparse operator
use crate::{
    space::{Element, IndexableSpace, LinearSpace},
    AsApply, OperatorBase,
};
use rlst_dense::traits::{RawAccess, RawAccessMut, Shape};
use rlst_dense::types::RlstScalar;
use rlst_sparse::sparse::csc_mat::CscMatrix;
use rlst_sparse::sparse::csr_mat::CsrMatrix;

use super::array_vector_space::ArrayVectorSpace;

/// CSR matrix operator
pub struct CsrMatrixOperator<'a, Item: RlstScalar> {
    csr_mat: &'a CsrMatrix<Item>,
    domain: &'a ArrayVectorSpace<Item>,
    range: &'a ArrayVectorSpace<Item>,
}

impl<Item: RlstScalar> std::fmt::Debug for CsrMatrixOperator<'_, Item> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CsrMatrixOperator")
            .field("Dimension", &self.csr_mat.shape())
            .field("Type", &"csr")
            .finish()
    }
}

impl<'a, Item: RlstScalar> CsrMatrixOperator<'a, Item> {
    /// Create a new CSR matrix operator
    pub fn new(
        csr_mat: &'a CsrMatrix<Item>,
        domain: &'a ArrayVectorSpace<Item>,
        range: &'a ArrayVectorSpace<Item>,
    ) -> Self {
        let shape = csr_mat.shape();
        assert_eq!(domain.dimension(), shape[1]);
        assert_eq!(range.dimension(), shape[0]);
        Self {
            csr_mat,
            domain,
            range,
        }
    }
}

impl<Item: RlstScalar> OperatorBase for CsrMatrixOperator<'_, Item> {
    type Domain = ArrayVectorSpace<Item>;

    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> &Self::Domain {
        self.domain
    }

    fn range(&self) -> &Self::Range {
        self.range
    }
}

impl<Item: RlstScalar> AsApply for CsrMatrixOperator<'_, Item> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> rlst_dense::types::RlstResult<()> {
        self.csr_mat
            .matmul(alpha, x.view().data(), beta, y.view_mut().data_mut());
        Ok(())
    }
}

/// CSC matrix operator
pub struct CscMatrixOperator<'a, Item: RlstScalar> {
    csc_mat: &'a CscMatrix<Item>,
    domain: &'a ArrayVectorSpace<Item>,
    range: &'a ArrayVectorSpace<Item>,
}

impl<Item: RlstScalar> std::fmt::Debug for CscMatrixOperator<'_, Item> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CscMatrixOperator")
            .field("Dimension", &self.csc_mat.shape())
            .field("Type", &"csc")
            .finish()
    }
}

impl<'a, Item: RlstScalar> CscMatrixOperator<'a, Item> {
    /// Create a new CSC matrix operator
    pub fn new(
        csc_mat: &'a CscMatrix<Item>,
        domain: &'a ArrayVectorSpace<Item>,
        range: &'a ArrayVectorSpace<Item>,
    ) -> Self {
        let shape = csc_mat.shape();
        assert_eq!(domain.dimension(), shape[1]);
        assert_eq!(range.dimension(), shape[0]);
        Self {
            csc_mat,
            domain,
            range,
        }
    }
}

impl<Item: RlstScalar> OperatorBase for CscMatrixOperator<'_, Item> {
    type Domain = ArrayVectorSpace<Item>;

    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> &Self::Domain {
        self.domain
    }

    fn range(&self) -> &Self::Range {
        self.range
    }
}

impl<Item: RlstScalar> AsApply for CscMatrixOperator<'_, Item> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> rlst_dense::types::RlstResult<()> {
        self.csc_mat
            .matmul(alpha, x.view().data(), beta, y.view_mut().data_mut());
        Ok(())
    }
}
