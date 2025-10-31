//! Matrix interface.
//!
//! This module defines interfaces to the Rlst dense and sparse matrix types
//! to be used as abstract operators.

#[cfg(feature = "mpi")]
use mpi::traits::{Communicator, Equivalence};

use crate::{
    Array, AsMatrixApply, BaseItem, LinearSpace, Shape, abstract_operator::OperatorBase,
    dense::array::DynArray, operator::abstract_operator::Operator, sparse::csr_mat::CsrMatrix,
};

#[cfg(feature = "mpi")]
use crate::{
    dense::{base_array::BaseArray, data_container::VectorContainer},
    operator::interface::distributed_array_vector_space::DistributedArrayVectorSpace,
    sparse::distributed_array::DistributedArray,
    sparse::distributed_csr_mat::DistributedCsrMatrix,
};

use super::array_vector_space::ArrayVectorSpace;

/// An [ArrayOperator] is defined by a two-dimensional array representing a matrix.
pub struct ArrayOperator<'a, ArrayImpl>
where
    ArrayImpl: BaseItem,
{
    op: &'a Array<ArrayImpl, 2>,
    domain: ArrayVectorSpace<ArrayImpl::Item>,
    range: ArrayVectorSpace<ArrayImpl::Item>,
}

impl<'a, ArrayImpl: BaseItem> ArrayOperator<'a, ArrayImpl> {
    /// Create a new `ArrayOperator` from a given 2-dimensional array.
    pub fn new(op: &'a Array<ArrayImpl, 2>) -> Self
    where
        ArrayImpl: Shape<2>,
    {
        let shape = op.shape();
        Self {
            op,
            domain: ArrayVectorSpace::new(shape[1]),
            range: ArrayVectorSpace::new(shape[0]),
        }
    }
}

impl<'a, ArrayImpl: BaseItem + Shape<2>> From<&'a Array<ArrayImpl, 2>>
    for Operator<ArrayOperator<'a, ArrayImpl>>
where
    ArrayOperator<'a, ArrayImpl>: OperatorBase,
{
    fn from(value: &'a Array<ArrayImpl, 2>) -> Self {
        Operator::new(ArrayOperator::new(value))
    }
}

impl<'a, ArrayImpl> OperatorBase for ArrayOperator<'a, ArrayImpl>
where
    ArrayImpl: BaseItem,
    ArrayVectorSpace<ArrayImpl::Item>:
        LinearSpace<F = ArrayImpl::Item, Impl = DynArray<ArrayImpl::Item, 1>>,
    Array<ArrayImpl, 2>: AsMatrixApply<DynArray<ArrayImpl::Item, 1>, DynArray<ArrayImpl::Item, 1>>
        + BaseItem<Item = ArrayImpl::Item>,
{
    type Domain = ArrayVectorSpace<ArrayImpl::Item>;

    type Range = ArrayVectorSpace<ArrayImpl::Item>;

    fn domain(&self) -> &Self::Domain {
        &self.domain
    }

    fn range(&self) -> &Self::Range {
        &self.range
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as crate::LinearSpace>::F,
        x: &crate::operator::element::Element<Self::Domain>,
        beta: <Self::Range as crate::LinearSpace>::F,
        y: &mut crate::operator::element::Element<Self::Range>,
    ) {
        self.op.apply(alpha, x.imp(), beta, y.imp_mut());
    }
}

/// A [CsrOperator] is defined by a sparse CSR matrix.
pub struct CsrOperator<'a, Item> {
    arr: &'a CsrMatrix<Item>,
    domain: ArrayVectorSpace<Item>,
    range: ArrayVectorSpace<Item>,
}

impl<'a, Item> CsrOperator<'a, Item> {
    /// Create a new `CsrOperator` from a given CsrMatrix.
    pub fn new(arr: &'a CsrMatrix<Item>) -> Self {
        let shape = arr.shape();
        Self {
            arr,
            domain: ArrayVectorSpace::new(shape[1]),
            range: ArrayVectorSpace::new(shape[0]),
        }
    }
}

impl<'a, Item> From<&'a CsrMatrix<Item>> for Operator<CsrOperator<'a, Item>>
where
    CsrOperator<'a, Item>: OperatorBase,
{
    fn from(value: &'a CsrMatrix<Item>) -> Self {
        Operator::new(CsrOperator::new(value))
    }
}

impl<'a, Item> OperatorBase for CsrOperator<'a, Item>
where
    ArrayVectorSpace<Item>: LinearSpace<F = Item, Impl = DynArray<Item, 1>>,
    CsrMatrix<Item>: AsMatrixApply<DynArray<Item, 1>, DynArray<Item, 1>> + BaseItem<Item = Item>,
{
    type Domain = ArrayVectorSpace<Item>;

    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> &Self::Domain {
        &self.domain
    }

    fn range(&self) -> &Self::Range {
        &self.range
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &crate::operator::element::Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut crate::operator::element::Element<Self::Range>,
    ) {
        self.arr.apply(alpha, x.imp(), beta, y.imp_mut());
    }
}

#[cfg(feature = "mpi")]
/// A [DistributedCsrOperator] is defined by a distributed sparse CSR matrix.
pub struct DistributedCsrOperator<'a, Item: Equivalence, C: Communicator> {
    op: &'a DistributedCsrMatrix<'a, Item, C>,
    domain: DistributedArrayVectorSpace<'a, C, Item>,
    range: DistributedArrayVectorSpace<'a, C, Item>,
}

#[cfg(feature = "mpi")]
impl<'a, Item: Copy + Equivalence, C: Communicator> DistributedCsrOperator<'a, Item, C> {
    /// Create a new `DistributedCsrOperator` from a given CsrMatrix.
    pub fn new(op: &'a DistributedCsrMatrix<'a, Item, C>) -> Self {
        Self {
            op,
            domain: DistributedArrayVectorSpace::new(op.domain_layout().clone()),
            range: DistributedArrayVectorSpace::new(op.range_layout().clone()),
        }
    }
}

#[cfg(feature = "mpi")]
impl<'a, Item: Copy + Equivalence, C: Communicator> From<&'a DistributedCsrMatrix<'a, Item, C>>
    for Operator<DistributedCsrOperator<'a, Item, C>>
where
    DistributedCsrOperator<'a, Item, C>: OperatorBase,
{
    fn from(value: &'a DistributedCsrMatrix<'a, Item, C>) -> Self {
        Operator::new(DistributedCsrOperator::new(value))
    }
}

#[cfg(feature = "mpi")]
impl<'a, Item: Equivalence, C: Communicator> OperatorBase for DistributedCsrOperator<'a, Item, C>
where
    DistributedArrayVectorSpace<'a, C, Item>: LinearSpace<
            F = Item,
            Impl = DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>,
        >,
    DistributedCsrMatrix<'a, Item, C>: AsMatrixApply<
            DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>,
            DistributedArray<'a, C, BaseArray<VectorContainer<Item>, 1>, 1>,
        > + BaseItem<Item = Item>,
{
    type Domain = DistributedArrayVectorSpace<'a, C, Item>;

    type Range = DistributedArrayVectorSpace<'a, C, Item>;

    fn domain(&self) -> &Self::Domain {
        &self.domain
    }

    fn range(&self) -> &Self::Range {
        &self.range
    }

    #[inline(always)]
    fn apply(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &crate::operator::element::Element<Self::Domain>,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut crate::operator::element::Element<Self::Range>,
    ) {
        self.op.apply(alpha, x.imp(), beta, y.imp_mut());
    }
}
