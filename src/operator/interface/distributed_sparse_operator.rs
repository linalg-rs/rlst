//! Distributed sparse operator
use bempp_distributed_tools::IndexLayout;
use mpi::traits::{Communicator, Equivalence};

use crate::dense::traits::Shape;
use crate::dense::types::RlstScalar;
use crate::DistributedCsrMatrix;
use crate::{
    operator::space::{Element, IndexableSpace, LinearSpace},
    operator::AsApply,
    operator::OperatorBase,
};

use super::DistributedArrayVectorSpace;

/// CSR matrix operator
pub struct DistributedCsrMatrixOperator<
    'a,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    Item: RlstScalar + Equivalence,
    C: Communicator,
> {
    csr_mat: &'a DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>,
    domain: &'a DistributedArrayVectorSpace<'a, DomainLayout, Item>,
    range: &'a DistributedArrayVectorSpace<'a, RangeLayout, Item>,
}

impl<
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > std::fmt::Debug for DistributedCsrMatrixOperator<'_, DomainLayout, RangeLayout, Item, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedCsrMatrixOperator")
            .field("Dimension", &self.csr_mat.shape())
            .field("Type", &"distributed csr")
            .finish()
    }
}

impl<
        'a,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > DistributedCsrMatrixOperator<'a, DomainLayout, RangeLayout, Item, C>
{
    /// Create a new CSR matrix operator
    pub fn new(
        csr_mat: &'a DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>,
        domain: &'a DistributedArrayVectorSpace<'a, DomainLayout, Item>,
        range: &'a DistributedArrayVectorSpace<'a, RangeLayout, Item>,
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

impl<
        'a,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > OperatorBase for DistributedCsrMatrixOperator<'a, DomainLayout, RangeLayout, Item, C>
{
    type Domain = DistributedArrayVectorSpace<'a, DomainLayout, Item>;

    type Range = DistributedArrayVectorSpace<'a, RangeLayout, Item>;

    fn domain(&self) -> &Self::Domain {
        self.domain
    }

    fn range(&self) -> &Self::Range {
        self.range
    }
}

impl<
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > AsApply for DistributedCsrMatrixOperator<'_, DomainLayout, RangeLayout, Item, C>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> crate::dense::types::RlstResult<()> {
        self.csr_mat.matmul(alpha, x.view(), beta, y.view_mut());
        Ok(())
    }
}
