//! Distributed sparse operator
use std::rc::Rc;

use mpi::traits::{Communicator, Equivalence};

use crate::dense::traits::Shape;
use crate::dense::types::RlstScalar;
use crate::operator::Operator;
use crate::DistributedCsrMatrix;
use crate::{
    operator::space::{ElementImpl, IndexableSpace, LinearSpace},
    operator::AsApply,
    operator::OperatorBase,
};

use super::DistributedArrayVectorSpace;

/// CSR matrix operator
pub struct DistributedCsrMatrixOperatorImpl<'a, Item: RlstScalar + Equivalence, C: Communicator> {
    csr_mat: DistributedCsrMatrix<'a, Item, C>,
    domain: Rc<DistributedArrayVectorSpace<'a, C, Item>>,
    range: Rc<DistributedArrayVectorSpace<'a, C, Item>>,
}

impl<Item: RlstScalar + Equivalence, C: Communicator> std::fmt::Debug
    for DistributedCsrMatrixOperatorImpl<'_, Item, C>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedCsrMatrixOperator")
            .field("Dimension", &self.csr_mat.shape())
            .field("Type", &"distributed csr")
            .finish()
    }
}

impl<'a, Item: RlstScalar + Equivalence, C: Communicator>
    DistributedCsrMatrixOperatorImpl<'a, Item, C>
{
    /// Create a new CSR matrix operator
    pub fn new(
        csr_mat: DistributedCsrMatrix<'a, Item, C>,
        domain: Rc<DistributedArrayVectorSpace<'a, C, Item>>,
        range: Rc<DistributedArrayVectorSpace<'a, C, Item>>,
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

impl<'a, Item: RlstScalar + Equivalence, C: Communicator> OperatorBase
    for DistributedCsrMatrixOperatorImpl<'a, Item, C>
{
    type Domain = DistributedArrayVectorSpace<'a, C, Item>;
    type Range = DistributedArrayVectorSpace<'a, C, Item>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<Item: RlstScalar + Equivalence, C: Communicator> AsApply
    for DistributedCsrMatrixOperatorImpl<'_, Item, C>
{
    fn apply_extended<
        ContainerIn: crate::ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: crate::ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: crate::Element<ContainerIn>,
        beta: <Self::Range as LinearSpace>::F,
        mut y: crate::Element<ContainerOut>,
    ) {
        self.csr_mat
            .matmul(alpha, x.imp().view(), beta, y.imp_mut().view_mut());
    }

    fn apply_extended_transpose<
        ContainerIn: crate::ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: crate::ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: crate::Element<ContainerIn>,
        beta: <Self::Range as LinearSpace>::F,
        mut y: crate::Element<ContainerOut>,
    ) {
        self.csr_mat
            .matmul_transpose(alpha, x.imp().view(), beta, y.imp_mut().view_mut());
    }
}

impl<'a, Item: RlstScalar + Equivalence, C: Communicator> From<DistributedCsrMatrix<'a, Item, C>>
    for Operator<DistributedCsrMatrixOperatorImpl<'a, Item, C>>
{
    fn from(csr_mat: DistributedCsrMatrix<'a, Item, C>) -> Self {
        let domain_layout = csr_mat.domain_layout();
        let range_layout = csr_mat.range_layout();
        let domain = DistributedArrayVectorSpace::from_index_layout(domain_layout.clone());
        let range = DistributedArrayVectorSpace::from_index_layout(range_layout.clone());
        Operator::new(DistributedCsrMatrixOperatorImpl::new(
            csr_mat, domain, range,
        ))
    }
}
