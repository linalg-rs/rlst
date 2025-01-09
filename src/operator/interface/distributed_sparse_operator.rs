//! Distributed sparse operator
use bempp_distributed_tools::IndexLayout;
use mpi::traits::{Communicator, Equivalence};

use crate::dense::traits::Shape;
use crate::dense::types::RlstScalar;
use crate::operator::Operator;
use crate::DistributedCsrMatrix;
use crate::{
    operator::space::{Element, IndexableSpace, LinearSpace},
    operator::AsApply,
    operator::OperatorBase,
};

use super::DistributedArrayVectorSpace;

/// CSR matrix operator
pub struct DistributedCsrMatrixOperatorImpl<
    'a,
    DomainLayout: IndexLayout<Comm = C>,
    RangeLayout: IndexLayout<Comm = C>,
    Item: RlstScalar + Equivalence,
    C: Communicator,
> {
    csr_mat: DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>,
    domain: DistributedArrayVectorSpace<'a, DomainLayout, Item>,
    range: DistributedArrayVectorSpace<'a, RangeLayout, Item>,
}

impl<
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > std::fmt::Debug for DistributedCsrMatrixOperatorImpl<'_, DomainLayout, RangeLayout, Item, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedCsrMatrixOperator")
            .field("Dimension", &self.csr_mat.shape())
            .field("Type", &"distributed csr")
            .finish()
    }
}

impl<
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > DistributedCsrMatrixOperatorImpl<DomainLayout, RangeLayout, Item, C>
{
    /// Create a new CSR matrix operator
    pub fn new(
        csr_mat: DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>,
        domain: DistributedArrayVectorSpace<'a, DomainLayout, Item>,
        range: DistributedArrayVectorSpace<'a, RangeLayout, Item>,
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

    // /// Return a reference to the contained CSR matrix.
    // pub fn inner(&self) -> &DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C> {
    //     &self.csr_mat
    // }

    // /// Move the contained CSR matrix out of the operator.
    // pub fn into_inner(self) -> DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C> {
    //     self.csr_mat
    // }
}

impl<
        'a,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > OperatorBase for DistributedCsrMatrixOperatorImpl<'a, DomainLayout, RangeLayout, Item, C>
{
    type Domain = DistributedArrayVectorSpace<'a, DomainLayout, Item>;
    type Range = DistributedArrayVectorSpace<'a, RangeLayout, Item>;

    fn domain(&self) -> &Self::Domain {
        &self.domain
    }

    fn range(&self) -> &Self::Range {
        &self.range
    }
}

impl<
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
    > AsApply for DistributedCsrMatrixOperatorImpl<'_, DomainLayout, RangeLayout, Item, C>
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

impl<
        'a,
        DomainLayout: IndexLayout<Comm = C>,
        RangeLayout: IndexLayout<Comm = C>,
        Item: RlstScalar + Equivalence,
        C: Communicator,
        OpImpl: OperatorBase<
            Domain = DistributedArrayVectorSpace<'a, DomainLayout, Item>,
            Range = DistributedArrayVectorSpace<'a, RangeLayout, Item>,
        >,
    > From<DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>> for Operator<OpImpl>
{
    fn from(csr_mat: DistributedCsrMatrix<'a, DomainLayout, RangeLayout, Item, C>) -> Self {
        let domain_layout = csr_mat.domain_layout();
        let range_layout = csr_mat.range_layout();
        let domain = DistributedArrayVectorSpace::new(domain_layout);
        let range = DistributedArrayVectorSpace::new(range_layout);
        Operator::new(DistributedCsrMatrixOperatorImpl::new(
            csr_mat, domain, range,
        ))
    }
}
