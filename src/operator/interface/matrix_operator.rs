//! Dense matrix operator
use std::rc::Rc;

use crate::dense::types::RlstScalar;
use crate::dense::{
    array::Array,
    traits::{AsOperatorApply, MultInto, RawAccess, Shape, Stride, UnsafeRandomAccessByValue},
};
use crate::operator::Operator;
use crate::{
    operator::space::{ElementImpl, LinearSpace},
    operator::AsApply,
    operator::OperatorBase,
};
use crate::{
    rlst_array_from_slice1, rlst_array_from_slice_mut1, CscMatrix, CsrMatrix, RawAccessMut,
};

use super::array_vector_space::ArrayVectorSpace;

/// Matrix operator
pub struct MatrixOperator<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> {
    op: Op,
    domain: Rc<ArrayVectorSpace<Item>>,
    range: Rc<ArrayVectorSpace<Item>>,
}

/// Matrix operator reference
pub struct MatrixOperatorRef<'a, Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> {
    op: &'a Op,
    domain: Rc<ArrayVectorSpace<Item>>,
    range: Rc<ArrayVectorSpace<Item>>,
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> std::fmt::Debug
    for MatrixOperator<Item, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MatrixOperator: [{}x{}]",
            self.op.shape()[0],
            self.op.shape()[1]
        )
        .unwrap();
        Ok(())
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> std::fmt::Debug
    for MatrixOperatorRef<'_, Item, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MatrixOperator: [{}x{}]",
            self.op.shape()[0],
            self.op.shape()[1]
        )
        .unwrap();
        Ok(())
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> From<Op>
    for Operator<MatrixOperator<Item, Op>>
{
    fn from(op: Op) -> Self {
        let shape = op.shape();
        let domain = ArrayVectorSpace::from_dimension(shape[1]);
        let range = ArrayVectorSpace::from_dimension(shape[0]);
        Operator::new(MatrixOperator { op, domain, range })
    }
}

impl<'a, Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> From<&'a Op>
    for Operator<MatrixOperatorRef<'a, Item, Op>>
{
    fn from(op: &'a Op) -> Self {
        let shape = op.shape();
        let domain = ArrayVectorSpace::from_dimension(shape[1]);
        let range = ArrayVectorSpace::from_dimension(shape[0]);
        Self::new(MatrixOperatorRef { op, domain, range })
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> OperatorBase
    for MatrixOperator<Item, Op>
{
    type Domain = ArrayVectorSpace<Item>;
    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> AsApply
    for MatrixOperator<Item, Op>
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
        self.op.apply_extended(
            alpha,
            x.imp().view().data(),
            beta,
            y.imp_mut().view_mut().data_mut(),
        );
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> OperatorBase
    for MatrixOperatorRef<'_, Item, Op>
{
    type Domain = ArrayVectorSpace<Item>;
    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

impl<Item: RlstScalar, Op: AsOperatorApply<Item = Item> + Shape<2>> AsApply
    for MatrixOperatorRef<'_, Item, Op>
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
        self.op.apply_extended(
            alpha,
            x.imp().view().data(),
            beta,
            y.imp_mut().view_mut().data_mut(),
        );
    }
}

// Matrix operator trait for dense matrices

        Item: RlstScalar,
        ArrayImpl: UnsafeRandomAccessByValue<2, Item = Item> + Shape<2> + RawAccess<Item = Item> + Stride<2>,
    > AsOperatorApply for Array<Item, ArrayImpl, 2>
{
    type Item = Item;

    fn apply_extended(
        &self,
        alpha: Self::Item,
        x: &[Self::Item],
        beta: Self::Item,
        y: &mut [Self::Item],
    ) {
        assert_eq!(self.shape()[1], x.len());
        assert_eq!(self.shape()[0], y.len());
        let x_arr = rlst_array_from_slice1!(x, [x.len()]);
        let mut y_arr = rlst_array_from_slice_mut1!(y, [y.len()]);

        y_arr.r_mut().mult_into(
            crate::TransMode::NoTrans,
            crate::TransMode::NoTrans,
            alpha,
            self.r(),
            x_arr.r(),
            beta,
        );
    }
}

// Matrix operator trait for CSR matrices
impl<Item: RlstScalar> AsOperatorApply for CsrMatrix<Item> {
    type Item = Item;

    fn apply_extended(
        &self,
        alpha: Self::Item,
        x: &[Self::Item],
        beta: Self::Item,
        y: &mut [Self::Item],
    ) {
        self.matmul(alpha, x, beta, y);
    }
}

// Matrix operator trait for CSC matrices
impl<Item: RlstScalar> AsOperatorApply for CscMatrix<Item> {
    type Item = Item;

    fn apply_extended(
        &self,
        alpha: Self::Item,
        x: &[Self::Item],
        beta: Self::Item,
        y: &mut [Self::Item],
    ) {
        self.matmul(alpha, x, beta, y);
    }
}
