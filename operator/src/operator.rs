//! Definition of a general linear operator.

use crate::{FieldType, LinearSpace};
use num::{One, Zero};
use rlst_common::types::*;
use std::fmt::Debug;

// A base operator trait.
pub trait OperatorBase: Debug {
    type Domain: LinearSpace;
    type Range: LinearSpace;

    fn domain(&self) -> &Self::Domain;

    fn range(&self) -> &Self::Range;

    fn as_ref_obj(&self) -> RlstOperatorReference<'_, Self>
    where
        Self: Sized,
    {
        RlstOperatorReference(self)
    }

    fn scale(self, alpha: <Self::Range as LinearSpace>::F) -> ScalarTimesOperator<Self>
    where
        Self: Sized,
    {
        ScalarTimesOperator(self, alpha)
    }

    fn sum<Op: OperatorBase<Domain = Self::Domain, Range = Self::Range> + Sized>(
        self,
        other: Op,
    ) -> OperatorSum<Self::Domain, Self::Range, Self, Op>
    where
        Self: Sized,
    {
        OperatorSum(self, other)
    }
}

/// Apply an operator as y -> alpha * Ax + beta y
pub trait AsApply: OperatorBase {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()>;

    fn apply(&self, x: &<Self::Domain as LinearSpace>::E) -> <Self::Range as LinearSpace>::E {
        let mut out = self.range().zero();

        self.apply_extended(
            <FieldType<Self::Range> as One>::one(),
            x,
            <FieldType<Self::Range> as Zero>::zero(),
            &mut out,
        )
        .unwrap();
        out
    }
}

pub struct RlstOperatorReference<'a, Op: OperatorBase>(&'a Op);

impl<Op: OperatorBase> std::fmt::Debug for RlstOperatorReference<'_, Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorReference").field(&self.0).finish()
    }
}

impl<Op: OperatorBase> OperatorBase for RlstOperatorReference<'_, Op> {
    type Domain = Op::Domain;

    type Range = Op::Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<Op: AsApply> AsApply for RlstOperatorReference<'_, Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(alpha, x, beta, y)
    }
}

pub struct OperatorSum<
    Domain: LinearSpace,
    Range: LinearSpace,
    Op1: OperatorBase<Domain = Domain, Range = Range>,
    Op2: OperatorBase<Domain = Domain, Range = Range>,
>(Op1, Op2);

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: OperatorBase<Domain = Domain, Range = Range>,
        Op2: OperatorBase<Domain = Domain, Range = Range>,
    > std::fmt::Debug for OperatorSum<Domain, Range, Op1, Op2>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&&self.0)
            .field(&&self.1)
            .finish()
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: OperatorBase<Domain = Domain, Range = Range>,
        Op2: OperatorBase<Domain = Domain, Range = Range>,
    > OperatorBase for OperatorSum<Domain, Range, Op1, Op2>
{
    type Domain = Domain;

    type Range = Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<
        Domain: LinearSpace,
        Range: LinearSpace,
        Op1: AsApply<Domain = Domain, Range = Range>,
        Op2: AsApply<Domain = Domain, Range = Range>,
    > AsApply for OperatorSum<Domain, Range, Op1, Op2>
{
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(alpha, x, beta, y)?;
        self.1
            .apply_extended(alpha, x, <<Self::Range as LinearSpace>::F as One>::one(), y)?;
        Ok(())
    }
}

pub struct ScalarTimesOperator<Op: OperatorBase>(Op, <Op::Range as LinearSpace>::F);

impl<Op: OperatorBase> std::fmt::Debug for ScalarTimesOperator<Op> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("OperatorSum")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<Op: OperatorBase> OperatorBase for ScalarTimesOperator<Op> {
    type Domain = Op::Domain;

    type Range = Op::Range;

    fn domain(&self) -> &Self::Domain {
        self.0.domain()
    }

    fn range(&self) -> &Self::Range {
        self.0.range()
    }
}

impl<Op: AsApply> AsApply for ScalarTimesOperator<Op> {
    fn apply_extended(
        &self,
        alpha: <Self::Range as LinearSpace>::F,
        x: &<Self::Domain as LinearSpace>::E,
        beta: <Self::Range as LinearSpace>::F,
        y: &mut <Self::Range as LinearSpace>::E,
    ) -> RlstResult<()> {
        self.0.apply_extended(self.1 * alpha, x, beta, y)?;
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use crate::{
        implementation::{
            array_vector_space::ArrayVectorSpace, dense_matrix_operator::DenseMatrixOperator,
        },
        AsApply, Element, LinearSpace, OperatorBase,
    };
    use rlst_dense::{assert_array_relative_eq, rlst_dynamic_array2, traits::*};

    #[test]
    fn test_operator_algebra() {
        let mut mat1 = rlst_dynamic_array2!(f64, [4, 3]);
        let mut mat2 = rlst_dynamic_array2!(f64, [4, 3]);

        let domain = ArrayVectorSpace::new(3);
        let range = ArrayVectorSpace::new(4);

        mat1.fill_from_seed_equally_distributed(0);
        mat2.fill_from_seed_equally_distributed(1);

        let op1 = DenseMatrixOperator::from(mat1, &domain, &range);
        let op2 = DenseMatrixOperator::from(mat2, &domain, &range);

        let mut x = domain.zero();
        let mut y = range.zero();
        let mut y_expected = range.zero();
        x.view_mut().fill_from_seed_equally_distributed(2);
        y.view_mut().fill_from_seed_equally_distributed(3);
        y_expected.view_mut().fill_from(y.view());

        op2.apply_extended(2.0, &x, 3.5, &mut y_expected).unwrap();
        op1.apply_extended(10.0, &x, 1.0, &mut y_expected).unwrap();

        let sum = op1.scale(5.0).sum(op2.as_ref_obj());

        sum.apply_extended(2.0, &x, 3.5, &mut y).unwrap();

        assert_array_relative_eq!(y.view(), y_expected.view(), 1E-12);
    }
}
